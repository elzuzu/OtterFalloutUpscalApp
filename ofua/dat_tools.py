from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os
import struct
import zlib
import time
from PIL import Image, ImagePalette
from PyQt6.QtWidgets import QApplication
from .workers import WorkerSignals


class PaletteCache:
    """Caches palettes to avoid reloading them."""
    _cache: Dict[str, ImagePalette.ImagePalette] = {}

    @classmethod
    def get_palette(cls, pal_path: Path) -> ImagePalette.ImagePalette:
        key = str(pal_path)
        if key not in cls._cache:
            with open(pal_path, 'rb') as f:
                data = f.read(768)
            pal_list = []
            for i in range(0, 768, 3):
                r, g, b = data[i:i+3]
                pal_list.extend([r * 4, g * 4, b * 4])
            while len(pal_list) < 768:
                pal_list.append(0)
            cls._cache[key] = ImagePalette.raw("RGB", bytes(pal_list))
        return cls._cache[key]


@dataclass
class DATEntry:
    filename: str
    normalized_path: str
    is_compressed: bool
    decompressed_size: int
    packed_size: int
    offset: int
    data: Optional[bytes] = None


@dataclass
class FRMFrame:
    width: int
    height: int
    pixel_data: bytes
    offset_x: int
    offset_y: int


@dataclass
class FRMData:
    version: int
    fps: int
    action_frame: int
    frames_per_direction: int
    num_directions: int = 6
    directions_data: List[List[FRMFrame]] = field(default_factory=lambda: [[] for _ in range(6)])
    palette: Optional[ImagePalette.ImagePalette] = None


@dataclass
class ProcessingStats:
    total_frms: int = 0
    processed_frms: int = 0
    failed_frms: int = 0
    start_time: float = 0.0

    def estimate_remaining_time(self) -> str:
        if self.processed_frms == 0:
            return "Calcul en cours..."
        elapsed = time.time() - self.start_time
        rate = self.processed_frms / elapsed
        remaining = (self.total_frms - self.processed_frms) / rate
        return f"{remaining/60:.1f} minutes"


class DatArchive:
    def __init__(self, dat_filepath: Path, signals: Optional[WorkerSignals] = None):
        self.filepath = dat_filepath
        self.entries: Dict[str, DATEntry] = {}
        self._signals = signals

    def _log(self, message: str):
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def load_entries(self) -> bool:
        self._log(f"Chargement des entrées depuis {self.filepath.name}...")
        try:
            with open(self.filepath, 'rb') as f:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                f.seek(file_size - 8)
                tree_size_bytes = f.read(4)
                if len(tree_size_bytes) < 4:
                    raise ValueError("Impossible de lire la taille de l'arbre du fichier DAT.")
                tree_size = struct.unpack('<I', tree_size_bytes)[0]
                dir_offset = file_size - 8 - tree_size
                f.seek(dir_offset)
                current_pos_in_tree = 0
                while current_pos_in_tree < tree_size:
                    name_length_bytes = f.read(4)
                    if not name_length_bytes:
                        break
                    current_pos_in_tree += 4
                    name_length = struct.unpack('<I', name_length_bytes)[0]
                    filename_bytes = f.read(name_length)
                    current_pos_in_tree += name_length
                    try:
                        filename = filename_bytes.decode('utf-8').replace('\\', '/')
                    except UnicodeDecodeError:
                        try:
                            filename = filename_bytes.decode('latin-1').replace('\\', '/')
                        except UnicodeDecodeError:
                            filename = filename_bytes.decode('cp1252', errors='replace').replace('\\', '/')
                    flags_byte = f.read(1)
                    current_pos_in_tree += 1
                    flags = struct.unpack('<B', flags_byte)[0]
                    is_compressed = (flags & 0x01) != 0
                    decompressed_size_bytes = f.read(4)
                    current_pos_in_tree += 4
                    decompressed_size = struct.unpack('<I', decompressed_size_bytes)[0]
                    packed_size_bytes = f.read(4)
                    current_pos_in_tree += 4
                    packed_size = struct.unpack('<I', packed_size_bytes)[0]
                    offset_bytes = f.read(4)
                    current_pos_in_tree += 4
                    offset = struct.unpack('<I', offset_bytes)[0]
                    normalized_path = filename.lower()
                    entry = DATEntry(
                        filename=filename,
                        normalized_path=normalized_path,
                        is_compressed=is_compressed,
                        decompressed_size=decompressed_size,
                        packed_size=packed_size,
                        offset=offset
                    )
                    self.entries[normalized_path] = entry
                self._log(f"{len(self.entries)} entrées chargées depuis {self.filepath.name}.")
        except FileNotFoundError:
            self._log(f"Erreur: Fichier DAT {self.filepath} non trouvé.")
            if self._signals:
                self._signals.error.emit(f"Fichier DAT {self.filepath} non trouvé.")
            return False
        except Exception as e:
            self._log(f"Erreur lors du chargement de {self.filepath.name}: {e}")
            if self._signals:
                self._signals.error.emit(f"Erreur lors du chargement de {self.filepath.name}: {e}")
            return False
        return True

    def extract_file(self, entry_path: str, output_dir: Path) -> Optional[Path]:
        normalized_entry_path = entry_path.lower().replace('\\', '/')
        if normalized_entry_path not in self.entries:
            self._log(f"Fichier {entry_path} non trouvé dans {self.filepath.name}.")
            return None
        entry = self.entries[normalized_entry_path]
        output_path = output_dir / entry.filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.filepath, 'rb') as f:
                f.seek(entry.offset)
                data = f.read(entry.packed_size)
            if entry.is_compressed:
                try:
                    decompressed_data = zlib.decompress(data)
                except zlib.error:
                    decompress_obj = zlib.decompressobj(-zlib.MAX_WBITS)
                    decompressed_data = decompress_obj.decompress(data)
                    decompressed_data += decompress_obj.flush()
            else:
                decompressed_data = data
            with open(output_path, 'wb') as out_f:
                out_f.write(decompressed_data)
            entry.data = decompressed_data
            return output_path
        except Exception as e:
            self._log(f"Erreur lors de l'extraction de {entry.filename}: {e}")
            if self._signals:
                self._signals.error.emit(f"Erreur extraction {entry.filename}: {e}")
            return None

    def extract_all_frms_and_pal(self, output_dir: Path, pal_filename="color.pal") -> Tuple[List[Path], Optional[Path]]:
        frm_files: List[Path] = []
        pal_file_path = None
        normalized_pal_search_paths = [pal_filename.lower(), f"data/{pal_filename.lower()}"]
        count = 0
        total_entries = len(self.entries)
        for entry_norm_path, entry in self.entries.items():
            count += 1
            if self._signals and count % 100 == 0:
                progress = int((count / total_entries) * 100)
                self._signals.progress.emit(progress)
                QApplication.processEvents()
            if entry_norm_path.endswith(".frm"):
                extracted_path = self.extract_file(entry.filename, output_dir)
                if extracted_path:
                    frm_files.append(extracted_path)
            if not pal_file_path and entry_norm_path in normalized_pal_search_paths:
                extracted_pal_path = self.extract_file(entry.filename, output_dir / Path(entry.filename).parent)
                if extracted_pal_path:
                    pal_file_path = extracted_pal_path
                    self._log(f"Fichier palette {pal_filename} trouvé et extrait: {pal_file_path}")
        if self._signals:
            self._signals.progress.emit(100)
        return frm_files, pal_file_path


class FRMConverter:
    def __init__(self, signals: Optional[WorkerSignals] = None):
        self.palette_data: Optional[bytes] = None
        self.pil_palette: Optional[ImagePalette.ImagePalette] = None
        self.pil_palette_image_path: Optional[Path] = None
        self._signals = signals

    def _log(self, message: str):
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def load_palette(self, pal_filepath: Path) -> bool:
        try:
            with open(pal_filepath, 'rb') as f:
                self.palette_data = f.read(768)
            if len(self.palette_data) != 768:
                self._log(f"Erreur: Taille de palette incorrecte pour {pal_filepath}.")
                return False
            pil_palette_list = []
            for i in range(0, 768, 3):
                r, g, b = self.palette_data[i:i+3]
                pil_palette_list.extend([r * 4, g * 4, b * 4])
            while len(pil_palette_list) < 768:
                pil_palette_list.append(0)
            self.pil_palette = ImagePalette.raw("RGB", bytes(pil_palette_list))
            self._log(f"Palette chargée depuis {pal_filepath}")
            return True
        except Exception as e:
            self._log(f"Erreur lors du chargement de la palette {pal_filepath}: {e}")
            if self._signals:
                self._signals.error.emit(f"Erreur chargement palette: {e}")
            return False

    def set_pil_palette_image_path(self, path: Path):
        self.pil_palette_image_path = path

    def _validate_frm_file(self, frm_path: Path) -> bool:
        """Validate that a FRM file looks valid."""
        try:
            with open(frm_path, 'rb') as f:
                if f.seek(0, os.SEEK_END) < 62:
                    return False
            return True
        except Exception:
            return False

    def frm_to_png(self, frm_filepath: Path, output_dir_png: Path) -> List[Path]:
        if not self.pil_palette:
            self._log("Erreur: Palette non chargée.")
            return []
        png_files: List[Path] = []
        try:
            with open(frm_filepath, 'rb') as f:
                version = struct.unpack('<I', f.read(4))[0]
                fps = struct.unpack('<H', f.read(2))[0]
                action_frame = struct.unpack('<H', f.read(2))[0]
                frames_per_direction = struct.unpack('<H', f.read(2))[0]
                shift_x_per_direction = [struct.unpack('<h', f.read(2))[0] for _ in range(6)]
                shift_y_per_direction = [struct.unpack('<h', f.read(2))[0] for _ in range(6)]
                frame_data_offsets_per_direction = [struct.unpack('<I', f.read(4))[0] for _ in range(6)]
                frm_base_name = frm_filepath.stem
                for dir_idx in range(6):
                    if frame_data_offsets_per_direction[dir_idx] == 0 and frames_per_direction > 0:
                        continue
                    f.seek(frame_data_offsets_per_direction[dir_idx])
                    for frame_idx in range(frames_per_direction):
                        try:
                            width = struct.unpack('<H', f.read(2))[0]
                            height = struct.unpack('<H', f.read(2))[0]
                            pixel_data_size = struct.unpack('<I', f.read(4))[0]
                            offset_x = struct.unpack('<h', f.read(2))[0]
                            offset_y = struct.unpack('<h', f.read(2))[0]
                        except struct.error:
                            break
                        if width == 0 or height == 0 or pixel_data_size == 0:
                            continue
                        pixel_data = f.read(pixel_data_size)
                        if len(pixel_data) != pixel_data_size:
                            continue
                        image = Image.frombytes('P', (width, height), pixel_data)
                        image.putpalette(self.pil_palette)
                        png_filename = f"{frm_base_name}_d{dir_idx}_f{frame_idx}.png"
                        output_png_path = output_dir_png / png_filename
                        output_png_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(output_png_path)
                        image.close()
                        del image
                        png_files.append(output_png_path)
            return png_files
        except Exception as e:
            self._log(f"Erreur lors de la conversion de {frm_filepath.name} en PNG: {e}")
            if self._signals:
                self._signals.error.emit(f"Erreur conversion {frm_filepath.name}: {e}")
            return []

    def png_to_frm(self, input_png_files_grouped: Dict[str, List[Path]], 
                   output_frm_dir: Path, original_frm_path: Path) -> Optional[Path]:
        """Convert upscaled PNG images back to a FRM file.

        Args:
            input_png_files_grouped: Dict mapping FRM base names to PNG file paths
            output_frm_dir: Directory where the resulting FRM should be stored
            original_frm_path: Path to the original FRM for metadata

        Returns:
            Path to the created FRM or None on failure
        """
        if not self.pil_palette:
            self._log("Erreur: Palette non chargée.")
            return None
        frm_base_name = original_frm_path.stem
        if frm_base_name not in input_png_files_grouped or not input_png_files_grouped[frm_base_name]:
            self._log(f"Aucun fichier PNG trouvé pour {frm_base_name}")
            return None
        png_files_for_frm = sorted(
            input_png_files_grouped[frm_base_name],
            key=lambda p: (int(p.stem.split('_d')[1].split('_f')[0]), int(p.stem.split('_f')[1]))
        )
        try:
            with open(original_frm_path, 'rb') as f_orig:
                original_version = struct.unpack('<I', f_orig.read(4))[0]
                original_fps = struct.unpack('<H', f_orig.read(2))[0]
                original_action_frame = struct.unpack('<H', f_orig.read(2))[0]
                original_frames_per_direction = struct.unpack('<H', f_orig.read(2))[0]
                original_shift_x = [struct.unpack('<h', f_orig.read(2))[0] for _ in range(6)]
                original_shift_y = [struct.unpack('<h', f_orig.read(2))[0] for _ in range(6)]
                original_frame_data_offsets = [struct.unpack('<I', f_orig.read(4))[0] for _ in range(6)]
        except Exception as e:
            self._log(f"Erreur lors de la lecture des métadonnées de {original_frm_path.name}: {e}")
            return None
        output_frm_path = output_frm_dir / original_frm_path.name
        output_frm_path.parent.mkdir(parents=True, exist_ok=True)
        frames_by_direction: List[List[tuple]] = [[] for _ in range(6)]
        max_frames_in_any_dir = 0
        for png_path in png_files_for_frm:
            try:
                parts = png_path.stem.split('_')
                dir_idx = int(parts[-2][1:])
                frame_idx = int(parts[-1][1:])
                img = Image.open(png_path)
                frames_by_direction[dir_idx].append((img, str(png_path), dir_idx, frame_idx))
                if frame_idx + 1 > max_frames_in_any_dir:
                    max_frames_in_any_dir = frame_idx + 1
            except Exception:
                continue
        actual_frames_per_direction = original_frames_per_direction or max_frames_in_any_dir
        frame_headers_data_bytes_by_dir = [bytearray() for _ in range(6)]
        all_frames_data_bytes = bytearray()
        new_frame_data_offsets = [0] * 6
        current_data_offset_from_header_end = 0
        for dir_idx in range(6):
            frames_in_this_dir = sorted(frames_by_direction[dir_idx], key=lambda x: x[3])
            if not frames_in_this_dir and original_frame_data_offsets[dir_idx] == 0 and actual_frames_per_direction > 0:
                for _ in range(actual_frames_per_direction):
                    frame_headers_data_bytes_by_dir[dir_idx] += struct.pack('<HHIsH', 0, 0, 0, 0, 0)
                new_frame_data_offsets[dir_idx] = 62 + current_data_offset_from_header_end
                current_data_offset_from_header_end += len(frame_headers_data_bytes_by_dir[dir_idx])
                continue
            if not frames_in_this_dir:
                new_frame_data_offsets[dir_idx] = 0
                continue
            new_frame_data_offsets[dir_idx] = 62 + current_data_offset_from_header_end
            for img, png_path_str, _, frame_idx_val in frames_in_this_dir:
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGBA')
                quantized_img = img.quantize(palette=Image.open(self.pil_palette_image_path), dither=Image.Dither.FLOYDSTEINBERG)
                pixel_data = quantized_img.tobytes()
                width, height = quantized_img.size
                pixel_data_size = len(pixel_data)
                offset_x, offset_y = 0, 0
                frame_headers_data_bytes_by_dir[dir_idx] += struct.pack('<HHIsH', width, height, pixel_data_size, offset_x, offset_y)
                all_frames_data_bytes += pixel_data
            current_data_offset_from_header_end += len(frame_headers_data_bytes_by_dir[dir_idx])
        try:
            with open(output_frm_path, 'wb') as f_out:
                f_out.write(struct.pack('<I', original_version))
                f_out.write(struct.pack('<H', original_fps))
                f_out.write(struct.pack('<H', original_action_frame))
                f_out.write(struct.pack('<H', actual_frames_per_direction))
                for sx in original_shift_x:
                    f_out.write(struct.pack('<h', sx))
                for sy in original_shift_y:
                    f_out.write(struct.pack('<h', sy))
                for offset in new_frame_data_offsets:
                    f_out.write(struct.pack('<I', offset))
                for dir_header_bytes in frame_headers_data_bytes_by_dir:
                    f_out.write(dir_header_bytes)
                f_out.write(all_frames_data_bytes)
            self._log(f"Fichier FRM {output_frm_path.name} créé.")
            return output_frm_path
        except Exception as e:
            self._log(f"Erreur lors de l'écriture du fichier FRM {output_frm_path.name}: {e}")
            if self._signals:
                self._signals.error.emit(f"Erreur écriture FRM {output_frm_path.name}: {e}")
            return None

