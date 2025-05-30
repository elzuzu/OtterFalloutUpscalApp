import sys
import os
import subprocess
import zlib
import struct
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImagePalette
import git # pip install GitPython

# Importation des modules PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QProgressBar, QLineEdit, QFileDialog,
    QLabel, QMessageBox, QGroupBox
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QRunnable, QThreadPool
from PyQt6.QtGui import QPalette, QColor

# --- Configuration ---
FALLOUT_CE_REPO_URL = "https://github.com/elzuzu/fallout1-ce.git"
DEFAULT_WORKSPACE_DIR = Path("./fallout_upscaler_workspace")
DEFAULT_REALESRGAN_EXE = "" # À remplir par l'utilisateur via l'UI
DEFAULT_REALESRGAN_MODEL = "" # À remplir par l'utilisateur via l'UI (nom du modèle sans extension)
DEFAULT_UPSCALE_FACTOR = "4" # Facteur d'upscale pour Real-ESRGAN

# --- Classes de Données ---
@dataclass
class DATEntry:
    """Représente une entrée dans un fichier DAT."""
    filename: str
    normalized_path: str # Chemin normalisé, ex: art/critters/myfile.frm
    is_compressed: bool
    decompressed_size: int
    packed_size: int
    offset: int
    data: Optional[bytes] = None

@dataclass
class FRMFrame:
    """Représente une frame dans un fichier FRM."""
    width: int
    height: int
    pixel_data: bytes # Données de pixels indexées
    offset_x: int
    offset_y: int

@dataclass
class FRMData:
    """Représente les données d'un fichier FRM."""
    version: int
    fps: int
    action_frame: int
    frames_per_direction: int
    num_directions: int = 6 # Fallout a typiquement 6 directions
    directions_data: List[List[FRMFrame]] = field(default_factory=lambda: [[] for _ in range(6)])
    palette: Optional[ImagePalette.ImagePalette] = None


# --- Signaux pour la communication entre threads et UI ---
class WorkerSignals(QObject):
    """Signaux pour un worker QThread/QRunnable."""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    request_config = pyqtSignal(str) # Pour demander des configurations comme le chemin de RealESRGAN

# --- Worker pour tâches en arrière-plan ---
class TaskRunner(QRunnable):
    """Exécute une fonction dans un thread séparé."""
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            self.fn(self.signals, *self.args, **self.kwargs)
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(f"Erreur dans le worker: {e}\n{traceback.format_exc()}")


# --- Gestionnaire de Fichiers DAT ---
class DatArchive:
    """Gère l'extraction des fichiers DAT de Fallout."""
    def __init__(self, dat_filepath: Path, signals: Optional[WorkerSignals] = None):
        self.filepath = dat_filepath
        self.entries: Dict[str, DATEntry] = {}
        self._signals = signals

    def _log(self, message: str):
        if self._signals:
            self.signals.log.emit(message)
        else:
            print(message)

    def load_entries(self):
        """Charge les entrées du fichier DAT."""
        self._log(f"Chargement des entrées depuis {self.filepath.name}...")
        try:
            with open(self.filepath, 'rb') as f:
                # La structure des DAT de Fallout (DAT2) est un peu particulière.
                # Le répertoire est à la fin du fichier.
                f.seek(0, os.SEEK_END)
                file_size = f.tell()

                f.seek(file_size - 8) # 4 octets pour tree_size, 4 pour file_count (pas toujours utilisé directement)
                
                # Lire la taille de l'arbre de répertoires (4 octets, little-endian)
                tree_size_bytes = f.read(4)
                if len(tree_size_bytes) < 4:
                    raise ValueError("Impossible de lire la taille de l'arbre du fichier DAT (fin de fichier inattendue).")
                tree_size = struct.unpack('<I', tree_size_bytes)[0]

                # Le nombre de fichiers est aussi parfois stocké ici, mais il est plus fiable de le compter
                # ou de lire jusqu'à la fin de la section de l'arbre.

                dir_offset = file_size - 8 - tree_size
                f.seek(dir_offset)
                
                current_pos_in_tree = 0
                while current_pos_in_tree < tree_size:
                    # Lire la longueur du nom de fichier (incluant le chemin)
                    name_length_bytes = f.read(4)
                    if not name_length_bytes: break # Fin de l'arbre
                    current_pos_in_tree += 4
                    name_length = struct.unpack('<I', name_length_bytes)[0]
                    
                    # Lire le nom de fichier
                    filename_bytes = f.read(name_length)
                    current_pos_in_tree += name_length
                    # Tenter de décoder en UTF-8, sinon en latin-1 ou cp1252
                    try:
                        filename = filename_bytes.decode('utf-8').replace('\\', '/')
                    except UnicodeDecodeError:
                        try:
                            filename = filename_bytes.decode('latin-1').replace('\\', '/')
                        except UnicodeDecodeError:
                            filename = filename_bytes.decode('cp1252', errors='replace').replace('\\', '/')


                    # Lire les flags (1 octet)
                    flags_byte = f.read(1)
                    current_pos_in_tree += 1
                    flags = struct.unpack('<B', flags_byte)[0]
                    is_compressed = (flags & 0x01) != 0 # Le bit 0 indique la compression

                    # Lire la taille décompressée (4 octets)
                    decompressed_size_bytes = f.read(4)
                    current_pos_in_tree += 4
                    decompressed_size = struct.unpack('<I', decompressed_size_bytes)[0]

                    # Lire la taille compressée (4 octets)
                    packed_size_bytes = f.read(4)
                    current_pos_in_tree += 4
                    packed_size = struct.unpack('<I', packed_size_bytes)[0]

                    # Lire l'offset des données du fichier (4 octets)
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
            if self._signals: self.signals.error.emit(f"Fichier DAT {self.filepath} non trouvé.")
            return False
        except Exception as e:
            self._log(f"Erreur lors du chargement de {self.filepath.name}: {e}")
            if self._signals: self.signals.error.emit(f"Erreur lors du chargement de {self.filepath.name}: {e}")
            import traceback
            self._log(traceback.format_exc())
            return False
        return True

    def extract_file(self, entry_path: str, output_dir: Path) -> Optional[Path]:
        """Extrait un fichier spécifique du DAT."""
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
                # La décompression zlib dans Fallout peut nécessiter un wbits négatif
                # ou pas d'en-tête zlib standard. Essayons avec zlib.decompress.
                try:
                    decompressed_data = zlib.decompress(data)
                except zlib.error as zde:
                    # Fallout utilise parfois un format zlib sans en-tête ni checksum.
                    # Essayez avec wbits = -15
                    try:
                        decompress_obj = zlib.decompressobj(-zlib.MAX_WBITS)
                        decompressed_data = decompress_obj.decompress(data)
                        decompressed_data += decompress_obj.flush()
                    except zlib.error as zde2:
                        self._log(f"Échec de la décompression ZLIB pour {entry.filename}: {zde} et {zde2}")
                        return None
            else:
                decompressed_data = data
            
            if len(decompressed_data) != entry.decompressed_size:
                self._log(f"Attention: Taille décompressée de {entry.filename} ({len(decompressed_data)}) ne correspond pas à la taille attendue ({entry.decompressed_size}).")


            with open(output_path, 'wb') as out_f:
                out_f.write(decompressed_data)
            # self._log(f"Fichier {entry.filename} extrait vers {output_path}")
            entry.data = decompressed_data # Stocker les données pour un accès ultérieur si nécessaire
            return output_path
        except Exception as e:
            self._log(f"Erreur lors de l'extraction de {entry.filename}: {e}")
            if self._signals: self.signals.error.emit(f"Erreur extraction {entry.filename}: {e}")
            return None

    def extract_all_frms_and_pal(self, output_dir: Path, pal_filename="color.pal") -> Tuple[List[Path], Optional[Path]]:
        """Extrait tous les fichiers FRM et le fichier palette spécifié."""
        frm_files = []
        pal_file_path = None
        
        # Normaliser le nom du fichier palette recherché
        normalized_pal_search_paths = [
            pal_filename.lower(),
            f"data/{pal_filename.lower()}" # Chemin commun pour color.pal
        ]

        count = 0
        total_entries = len(self.entries)
        for entry_norm_path, entry in self.entries.items():
            count +=1
            if self._signals and count % 100 == 0 : # Mettre à jour la progression moins fréquemment
                 progress = int((count / total_entries) * 100)
                 self._signals.progress.emit(progress)
                 QApplication.processEvents() # Permet à l'UI de se rafraîchir

            if entry_norm_path.endswith(".frm"):
                extracted_path = self.extract_file(entry.filename, output_dir)
                if extracted_path:
                    frm_files.append(extracted_path)
            
            # Chercher le fichier palette
            if not pal_file_path and entry_norm_path in normalized_pal_search_paths:
                extracted_pal_path = self.extract_file(entry.filename, output_dir / Path(entry.filename).parent) # Extraire dans son sous-dossier
                if extracted_pal_path:
                    pal_file_path = extracted_pal_path
                    self._log(f"Fichier palette {pal_filename} trouvé et extrait: {pal_file_path}")

        if self._signals: self._signals.progress.emit(100)
        return frm_files, pal_file_path


# --- Convertisseur FRM <-> PNG ---
class FRMConverter:
    """Convertit les fichiers FRM de Fallout en PNG et vice-versa."""
    def __init__(self, signals: Optional[WorkerSignals] = None):
        self.palette_data: Optional[bytes] = None
        self.pil_palette: Optional[ImagePalette.ImagePalette] = None
        self._signals = signals

    def _log(self, message: str):
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def load_palette(self, pal_filepath: Path) -> bool:
        """Charge la palette depuis un fichier .PAL."""
        try:
            with open(pal_filepath, 'rb') as f:
                self.palette_data = f.read(768) # 256 couleurs * 3 octets (RGB)
            if len(self.palette_data) != 768:
                self._log(f"Erreur: Taille de palette incorrecte pour {pal_filepath}. Attendu 768 octets, obtenu {len(self.palette_data)}.")
                return False

            # Convertir les données de palette (0-63) en format PIL (0-255)
            pil_palette_list = []
            for i in range(0, 768, 3):
                r, g, b = self.palette_data[i:i+3]
                pil_palette_list.extend([r * 4, g * 4, b * 4]) # Multiplier par 4
            
            # S'assurer que la liste a 768 éléments pour PIL
            while len(pil_palette_list) < 768:
                pil_palette_list.append(0) # Remplir avec du noir si nécessaire

            self.pil_palette = ImagePalette.raw("RGB", bytes(pil_palette_list))
            self._log(f"Palette chargée depuis {pal_filepath}")
            return True
        except Exception as e:
            self._log(f"Erreur lors du chargement de la palette {pal_filepath}: {e}")
            if self._signals: self._signals.error.emit(f"Erreur chargement palette: {e}")
            return False

    def frm_to_png(self, frm_filepath: Path, output_dir_png: Path) -> List[Path]:
        """Convertit un fichier FRM en une ou plusieurs images PNG."""
        if not self.pil_palette:
            self._log("Erreur: Palette non chargée. Impossible de convertir FRM en PNG.")
            return []

        png_files = []
        try:
            with open(frm_filepath, 'rb') as f:
                # Lire l'en-tête FRM (14 octets)
                version = struct.unpack('<I', f.read(4))[0]
                fps = struct.unpack('<H', f.read(2))[0]
                action_frame = struct.unpack('<H', f.read(2))[0]
                frames_per_direction = struct.unpack('<H', f.read(2))[0]
                
                # Décalages pour chaque direction (6 directions * 2 octets)
                shift_x_per_direction = [struct.unpack('<h', f.read(2))[0] for _ in range(6)]
                shift_y_per_direction = [struct.unpack('<h', f.read(2))[0] for _ in range(6)]
                
                # Offset des données de frame pour chaque direction (6 directions * 4 octets)
                # Certains FRM peuvent avoir 0 ici si pas de frame pour cette direction
                frame_data_offsets_per_direction = []
                for _ in range(6):
                    offset_val = struct.unpack('<I', f.read(4))[0]
                    frame_data_offsets_per_direction.append(offset_val)

                # Taille totale de l'en-tête (variable, mais au moins 14 + 12 + 12 + 24 = 62 octets)
                # Le reste du fichier contient les données des frames.

                frm_base_name = frm_filepath.stem
                
                for dir_idx in range(6): # Pour chaque direction
                    if frame_data_offsets_per_direction[dir_idx] == 0 and frames_per_direction > 0 :
                        # self._log(f"  Direction {dir_idx}: Pas de données de frame (offset 0).")
                        continue # Sauter si pas de données pour cette direction

                    # Se positionner au début des données de frame pour cette direction
                    # Note: les offsets sont relatifs au début du fichier
                    f.seek(frame_data_offsets_per_direction[dir_idx])
                    
                    for frame_idx in range(frames_per_direction):
                        # Lire l'en-tête de la frame (12 octets)
                        try:
                            width = struct.unpack('<H', f.read(2))[0]
                            height = struct.unpack('<H', f.read(2))[0]
                            pixel_data_size = struct.unpack('<I', f.read(4))[0] # Taille des données pixel
                            offset_x = struct.unpack('<h', f.read(2))[0]
                            offset_y = struct.unpack('<h', f.read(2))[0]
                        except struct.error:
                            # self._log(f"Fin prématurée des données de frame pour {frm_base_name} dir {dir_idx} frame {frame_idx}")
                            break # Sortir de la boucle des frames si on ne peut pas lire l'en-tête

                        if width == 0 or height == 0 or pixel_data_size == 0:
                            # self._log(f"  Frame {frame_idx} (Dir {dir_idx}): Dimensions/taille nulles, sautée.")
                            continue
                        
                        # Lire les données de pixels (indexées)
                        pixel_data = f.read(pixel_data_size)
                        if len(pixel_data) != pixel_data_size:
                            # self._log(f"Erreur: Données de pixel incomplètes pour {frm_base_name} dir {dir_idx} frame {frame_idx}. Attendu {pixel_data_size}, obtenu {len(pixel_data)}")
                            continue
                        
                        # Créer l'image Pillow
                        image = Image.frombytes('P', (width, height), pixel_data)
                        image.putpalette(self.pil_palette)
                        
                        # Sauvegarder en PNG
                        # Utiliser un nom de fichier qui inclut le nom original, la direction et la frame
                        # ex: MYFRM_d0_f0.png
                        png_filename = f"{frm_base_name}_d{dir_idx}_f{frame_idx}.png"
                        output_png_path = output_dir_png / png_filename
                        output_png_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(output_png_path)
                        png_files.append(output_png_path)
            # self._log(f"{len(png_files)} frames PNG créées pour {frm_filepath.name}")
            return png_files

        except FileNotFoundError:
            self._log(f"Erreur: Fichier FRM {frm_filepath} non trouvé.")
            if self._signals: self._signals.error.emit(f"Fichier FRM {frm_filepath} non trouvé.")
            return []
        except Exception as e:
            self._log(f"Erreur lors de la conversion de {frm_filepath.name} en PNG: {e}")
            if self._signals: self._signals.error.emit(f"Erreur conversion {frm_filepath.name}: {e}")
            import traceback
            self._log(traceback.format_exc())
            return []

    def png_to_frm(self, input_png_files_grouped: Dict[str, List[Path]], output_frm_dir: Path, original_frm_path: Path) -> Optional[Path]:
        """
        Convertit un groupe d'images PNG upscalées en un fichier FRM.
        input_png_files_grouped: Dictionnaire {frm_base_name: [path_to_png_d0_f0, path_to_png_d0_f1, ...]}
        original_frm_path: Chemin vers le fichier FRM original pour lire les métadonnées (FPS, ActionFrame, etc.)
        """
        if not self.pil_palette:
            self._log("Erreur: Palette non chargée. Impossible de convertir PNG en FRM.")
            return None

        frm_base_name = original_frm_path.stem
        if frm_base_name not in input_png_files_grouped or not input_png_files_grouped[frm_base_name]:
            self._log(f"Aucun fichier PNG trouvé pour {frm_base_name}")
            return None

        png_files_for_frm = sorted(input_png_files_grouped[frm_base_name], key=lambda p: (
            int(p.stem.split('_d')[1].split('_f')[0]), # dir_idx
            int(p.stem.split('_f')[1]) # frame_idx
        ))

        # Lire les métadonnées du FRM original
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

        # Organiser les frames PNG par direction
        frames_by_direction: List[List[Tuple[Image.Image, str, int, int]]] = [[] for _ in range(6)] # (image, png_path, dir_idx, frame_idx)
        max_frames_in_any_dir = 0

        for png_path in png_files_for_frm:
            try:
                parts = png_path.stem.split('_')
                dir_idx = int(parts[-2][1:]) # dX
                frame_idx = int(parts[-1][1:]) # fX
                
                img = Image.open(png_path)
                frames_by_direction[dir_idx].append((img, str(png_path), dir_idx, frame_idx))
                if frame_idx + 1 > max_frames_in_any_dir:
                    max_frames_in_any_dir = frame_idx + 1
            except Exception as e:
                self._log(f"Erreur de traitement du nom de fichier PNG {png_path.name} ou chargement: {e}")
                continue
        
        # S'assurer que frames_per_direction est correct
        # Si le FRM original avait 0 frames/dir, mais qu'on a des PNGs, il faut ajuster.
        # Pour l'instant, on utilise original_frames_per_direction, mais cela pourrait être affiné.
        # Si original_frames_per_direction est 0, mais on a des frames, on utilise max_frames_in_any_dir
        actual_frames_per_direction = original_frames_per_direction if original_frames_per_direction > 0 else max_frames_in_any_dir
        if actual_frames_per_direction == 0 and max_frames_in_any_dir > 0: # Cas où FRM original était vide mais on a des frames
             actual_frames_per_direction = max_frames_in_any_dir


        all_frames_data_bytes = bytearray()
        frame_headers_data_bytes_by_dir: List[bytearray] = [bytearray() for _ in range(6)]
        new_frame_data_offsets = [0] * 6

        # Header FRM: 14 octets (version, fps, action_frame, frames_per_direction)
        # + 6*2 (shift_x) + 6*2 (shift_y) + 6*4 (frame_data_offsets) = 14 + 12 + 12 + 24 = 62 octets
        current_data_offset_from_header_end = 0

        for dir_idx in range(6):
            frames_in_this_dir = sorted(frames_by_direction[dir_idx], key=lambda x: x[3]) # Trier par frame_idx
            
            if not frames_in_this_dir and original_frame_data_offsets[dir_idx] == 0 and actual_frames_per_direction > 0:
                # Si le FRM original n'avait pas de données pour cette direction,
                # mais que le FRM doit avoir des frames (actual_frames_per_direction > 0),
                # on doit quand même écrire les en-têtes de frame vides.
                # Ceci est important si certaines directions sont vides mais d'autres non.
                for _ in range(actual_frames_per_direction):
                    frame_headers_data_bytes_by_dir[dir_idx] += struct.pack('<HHIsH', 0, 0, 0, 0, 0) # w, h, size, ox, oy
                new_frame_data_offsets[dir_idx] = 62 + current_data_offset_from_header_end # Offset après l'en-tête principal
                current_data_offset_from_header_end += len(frame_headers_data_bytes_by_dir[dir_idx])
                continue # Passe à la direction suivante

            if not frames_in_this_dir:
                new_frame_data_offsets[dir_idx] = 0 # Pas de données pour cette direction
                continue

            new_frame_data_offsets[dir_idx] = 62 + current_data_offset_from_header_end

            for img, png_path_str, _, frame_idx_val in frames_in_this_dir:
                # Quantifier l'image avec la palette Fallout
                # S'assurer que l'image est en mode RGB ou RGBA avant de la quantifier vers 'P'
                if img.mode not in ['RGB', 'RGBA']:
                    img = img.convert('RGBA') # Convertir en RGBA pour une meilleure quantification
                
                quantized_img = img.quantize(palette=Image.open(self.pil_palette_image_path), dither=Image.Dither.FLOYDSTEINBERG) # Utiliser un PIL.Image pour la palette
                
                pixel_data = quantized_img.tobytes()
                width, height = quantized_img.size
                pixel_data_size = len(pixel_data)
                
                # Les offsets X, Y sont généralement relatifs au "point chaud" du sprite.
                # Pour l'upscaling, on pourrait les recalculer ou les mettre à 0.
                # Pour l'instant, mettons-les à 0.
                offset_x, offset_y = 0, 0 
                
                # En-tête de frame (12 octets)
                frame_headers_data_bytes_by_dir[dir_idx] += struct.pack('<HHIsH', width, height, pixel_data_size, offset_x, offset_y)
                all_frames_data_bytes += pixel_data
            
            current_data_offset_from_header_end += len(frame_headers_data_bytes_by_dir[dir_idx])


        try:
            with open(output_frm_path, 'wb') as f_out:
                # Écrire l'en-tête FRM principal
                f_out.write(struct.pack('<I', original_version)) # Version
                f_out.write(struct.pack('<H', original_fps))     # FPS
                f_out.write(struct.pack('<H', original_action_frame)) # Action Frame
                f_out.write(struct.pack('<H', actual_frames_per_direction)) # Frames per Direction

                # Écrire les shifts (utiliser ceux de l'original pour l'instant)
                for sx in original_shift_x: f_out.write(struct.pack('<h', sx))
                for sy in original_shift_y: f_out.write(struct.pack('<h', sy))

                # Écrire les nouveaux offsets des données de frame
                for offset in new_frame_data_offsets: f_out.write(struct.pack('<I', offset))

                # Écrire tous les en-têtes de frame concaténés
                for dir_header_bytes in frame_headers_data_bytes_by_dir:
                    f_out.write(dir_header_bytes)
                
                # Écrire toutes les données de pixel concaténées
                f_out.write(all_frames_data_bytes)
            
            self._log(f"Fichier FRM {output_frm_path.name} créé.")
            return output_frm_path

        except Exception as e:
            self._log(f"Erreur lors de l'écriture du fichier FRM {output_frm_path.name}: {e}")
            if self._signals: self._signals.error.emit(f"Erreur écriture FRM {output_frm_path.name}: {e}")
            import traceback
            self._log(traceback.format_exc())
            return None

    def set_pil_palette_image_path(self, path: Path):
        """ Utilisé pour la quantification PNG->FRM, car quantize a besoin d'une image de palette. """
        self.pil_palette_image_path = path


# --- Upscaler IA (via CLI Real-ESRGAN) ---
class AIScaler:
    """Gère l'upscaling des images via un outil CLI."""
    def __init__(self, realesrgan_exe: Path, model_name: str, scale_factor: str, signals: Optional[WorkerSignals] = None):
        self.realesrgan_exe = Path(realesrgan_exe)
        self.model_name = model_name
        self.scale_factor = scale_factor
        self._signals = signals

    def _log(self, message: str):
        if self._signals:
            self._signals.log.emit(message)
        else:
            print(message)

    def upscale_directory(self, input_dir_png: Path, output_dir_upscaled_png: Path) -> bool:
        """Upscale toutes les images PNG d'un répertoire."""
        if not self.realesrgan_exe or not self.realesrgan_exe.exists():
            self._log(f"Erreur: Exécutable Real-ESRGAN non trouvé à {self.realesrgan_exe}")
            if self._signals: self._signals.error.emit("Real-ESRGAN non trouvé. Veuillez configurer le chemin.")
            return False
        
        # Vérifier si le modèle .param et .bin existent (Real-ESRGAN les cherche à côté de l'exe ou dans un sous-dossier 'models')
        model_param_path = self.realesrgan_exe.parent / f"{self.model_name}.param"
        model_bin_path = self.realesrgan_exe.parent / f"{self.model_name}.bin"
        # Alternativement, ils peuvent être dans un sous-dossier 'models'
        alt_model_param_path = self.realesrgan_exe.parent / "models" / f"{self.model_name}.param"
        alt_model_bin_path = self.realesrgan_exe.parent / "models" / f"{self.model_name}.bin"

        if not ( (model_param_path.exists() and model_bin_path.exists()) or \
                 (alt_model_param_path.exists() and alt_model_bin_path.exists()) ):
            self._log(f"Erreur: Fichiers modèle Real-ESRGAN (.param et .bin) pour '{self.model_name}' non trouvés.")
            self._log(f"Cherché dans: {self.realesrgan_exe.parent} et {self.realesrgan_exe.parent / 'models'}")
            if self._signals: self._signals.error.emit(f"Modèle '{self.model_name}' non trouvé.")
            return False

        output_dir_upscaled_png.mkdir(parents=True, exist_ok=True)

        command = [
            str(self.realesrgan_exe),
            "-i", str(input_dir_png),
            "-o", str(output_dir_upscaled_png),
            "-n", self.model_name,
            "-s", self.scale_factor,
            "-f", "png" # Spécifier le format de sortie
        ]
        
        self._log(f"Lancement de Real-ESRGAN: {' '.join(command)}")
        
        try:
            # Utiliser Popen pour une meilleure gestion des logs en temps réel si nécessaire
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            
            # Afficher la sortie de Real-ESRGAN en temps réel
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    self._log(f"[Real-ESRGAN] {line.strip()}")
                    if self._signals: QApplication.processEvents() # Garder l'UI réactive
            
            process.wait() # Attendre la fin du processus
            
            if process.returncode != 0:
                self._log(f"Erreur Real-ESRGAN (code de retour {process.returncode}).")
                # stderr a été redirigé vers stdout, donc déjà loggé.
                if self._signals: self._signals.error.emit(f"Real-ESRGAN a échoué (code {process.returncode}).")
                return False
            
            self._log("Upscaling Real-ESRGAN terminé.")
            return True
        except FileNotFoundError:
            self._log(f"Erreur: Exécutable Real-ESRGAN '{self.realesrgan_exe}' non trouvé ou non exécutable.")
            if self._signals: self._signals.error.emit("Real-ESRGAN non trouvé ou non exécutable.")
            return False
        except Exception as e:
            self._log(f"Erreur lors de l'exécution de Real-ESRGAN: {e}")
            if self._signals: self._signals.error.emit(f"Erreur exécution Real-ESRGAN: {e}")
            return False

# --- Application Principale PyQt6 ---
class FalloutUpscalerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fallout 1 CE Asset Upscaler")
        self.setGeometry(100, 100, 800, 700) # x, y, width, height

        # Configuration
        self.workspace_dir = DEFAULT_WORKSPACE_DIR
        self.realesrgan_exe_path: Optional[Path] = None
        self.realesrgan_model_name: Optional[str] = None
        self.upscale_factor: str = DEFAULT_UPSCALE_FACTOR
        self.fallout_ce_cloned_path: Optional[Path] = None
        self.color_pal_path: Optional[Path] = None
        self.pil_palette_image_for_quantize: Optional[Path] = None


        # Initialiser les chemins par défaut si possible
        if DEFAULT_REALESRGAN_EXE: self.realesrgan_exe_path = Path(DEFAULT_REALESRGAN_EXE)
        if DEFAULT_REALESRGAN_MODEL: self.realesrgan_model_name = DEFAULT_REALESRGAN_MODEL


        self.thread_pool = QThreadPool()
        self.log_messages = [] # Pour stocker les logs

        self._init_ui()
        self._load_settings() # Charger les chemins sauvegardés

    def _init_ui(self):
        """Initialise l'interface utilisateur."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Groupe Configuration ---
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()

        # Workspace
        ws_layout = QHBoxLayout()
        ws_layout.addWidget(QLabel("Répertoire de travail:"))
        self.workspace_edit = QLineEdit(str(self.workspace_dir))
        self.workspace_button = QPushButton("Choisir...")
        self.workspace_button.clicked.connect(self._select_workspace_dir)
        ws_layout.addWidget(self.workspace_edit)
        ws_layout.addWidget(self.workspace_button)
        config_layout.addLayout(ws_layout)

        # Real-ESRGAN Executable
        re_exe_layout = QHBoxLayout()
        re_exe_layout.addWidget(QLabel("Real-ESRGAN (.exe):"))
        self.realesrgan_exe_edit = QLineEdit(str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "")
        self.realesrgan_exe_button = QPushButton("Choisir...")
        self.realesrgan_exe_button.clicked.connect(self._select_realesrgan_exe)
        re_exe_layout.addWidget(self.realesrgan_exe_edit)
        re_exe_layout.addWidget(self.realesrgan_exe_button)
        config_layout.addLayout(re_exe_layout)

        # Real-ESRGAN Model
        re_model_layout = QHBoxLayout()
        re_model_layout.addWidget(QLabel("Modèle Real-ESRGAN (nom):"))
        self.realesrgan_model_edit = QLineEdit(self.realesrgan_model_name if self.realesrgan_model_name else "")
        re_model_layout.addWidget(self.realesrgan_model_edit)
        # Pourrait ajouter un bouton pour choisir le modèle si les fichiers .param/.bin sont séparés
        config_layout.addLayout(re_model_layout)
        
        # Upscale Factor
        us_factor_layout = QHBoxLayout()
        us_factor_layout.addWidget(QLabel("Facteur d'Upscale:"))
        self.upscale_factor_edit = QLineEdit(self.upscale_factor)
        us_factor_layout.addWidget(self.upscale_factor_edit)
        config_layout.addLayout(us_factor_layout)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # --- Groupe Actions ---
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout()
        self.start_button = QPushButton("Démarrer le Processus Complet")
        self.start_button.clicked.connect(self._start_full_process)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; }")
        actions_layout.addWidget(self.start_button)
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        # --- Logs ---
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("QTextEdit { background-color: #f0f0f0; border: 1px solid #ccc; }")
        log_layout.addWidget(self.log_text_edit)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        self.log_message("Application initialisée. Configurez les chemins et démarrez le processus.")

    def _select_workspace_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Choisir le répertoire de travail", str(self.workspace_dir))
        if directory:
            self.workspace_dir = Path(directory)
            self.workspace_edit.setText(str(self.workspace_dir))
            self._save_settings()

    def _select_realesrgan_exe(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Choisir l'exécutable Real-ESRGAN", "", "Exécutables (*.exe);;Tous les fichiers (*)")
        if filepath:
            self.realesrgan_exe_path = Path(filepath)
            self.realesrgan_exe_edit.setText(str(self.realesrgan_exe_path))
            self._save_settings()

    def _update_config_from_ui(self):
        """Met à jour la configuration interne à partir des champs de l'UI."""
        self.workspace_dir = Path(self.workspace_edit.text())
        self.realesrgan_exe_path = Path(self.realesrgan_exe_edit.text()) if self.realesrgan_exe_edit.text() else None
        self.realesrgan_model_name = self.realesrgan_model_edit.text() if self.realesrgan_model_edit.text() else None
        self.upscale_factor = self.upscale_factor_edit.text()
        self._save_settings() # Sauvegarder à chaque mise à jour pour persistance

    def _save_settings(self):
        """Sauvegarde les chemins de configuration."""
        settings = {
            "workspace_dir": str(self.workspace_dir),
            "realesrgan_exe_path": str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "",
            "realesrgan_model_name": self.realesrgan_model_name if self.realesrgan_model_name else "",
            "upscale_factor": self.upscale_factor
        }
        try:
            config_path = Path.home() / ".fallout_upscaler_settings.json"
            with open(config_path, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            self.log_message(f"Avertissement: Impossible de sauvegarder les paramètres: {e}")


    def _load_settings(self):
        """Charge les chemins de configuration sauvegardés."""
        try:
            config_path = Path.home() / ".fallout_upscaler_settings.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                self.workspace_dir = Path(settings.get("workspace_dir", str(DEFAULT_WORKSPACE_DIR)))
                re_exe_p = settings.get("realesrgan_exe_path")
                self.realesrgan_exe_path = Path(re_exe_p) if re_exe_p else None
                self.realesrgan_model_name = settings.get("realesrgan_model_name")
                self.upscale_factor = settings.get("upscale_factor", DEFAULT_UPSCALE_FACTOR)

                # Mettre à jour l'UI
                self.workspace_edit.setText(str(self.workspace_dir))
                self.realesrgan_exe_edit.setText(str(self.realesrgan_exe_path) if self.realesrgan_exe_path else "")
                self.realesrgan_model_edit.setText(self.realesrgan_model_name if self.realesrgan_model_name else "")
                self.upscale_factor_edit.setText(self.upscale_factor)
                self.log_message("Paramètres chargés.")
        except Exception as e:
            self.log_message(f"Avertissement: Impossible de charger les paramètres: {e}. Utilisation des valeurs par défaut.")


    @pyqtSlot(str)
    def log_message(self, message: str):
        """Affiche un message dans la zone de log."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{now}] {message}"
        self.log_messages.append(full_message)
        self.log_text_edit.append(full_message)
        self.log_text_edit.ensureCursorVisible() # Auto-scroll

    @pyqtSlot(int)
    def update_progress(self, value: int):
        """Met à jour la barre de progression."""
        self.progress_bar.setValue(value)

    @pyqtSlot(str)
    def handle_error(self, error_message: str):
        """Gère une erreur signalée par un worker."""
        self.log_message(f"ERREUR: {error_message}")
        QMessageBox.critical(self, "Erreur", error_message)
        self.start_button.setEnabled(True) # Réactiver le bouton

    @pyqtSlot()
    def on_pipeline_finished(self):
        """Appelé lorsque le pipeline complet est terminé."""
        self.log_message("Processus complet terminé.")
        QMessageBox.information(self, "Terminé", "Le processus d'upscaling est terminé.")
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100) # Ou 0 si on veut la réinitialiser


    def _start_full_process(self):
        """Lance le pipeline complet dans un thread séparé."""
        self._update_config_from_ui() # S'assurer que la config est à jour

        if not self.workspace_dir:
            QMessageBox.warning(self, "Configuration", "Veuillez définir un répertoire de travail.")
            return
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        if not self.realesrgan_exe_path or not self.realesrgan_exe_path.exists():
            QMessageBox.warning(self, "Configuration", "Chemin vers Real-ESRGAN (.exe) non valide.")
            return
        if not self.realesrgan_model_name:
            QMessageBox.warning(self, "Configuration", "Nom du modèle Real-ESRGAN manquant.")
            return
        if not self.upscale_factor.isdigit() or not 1 <= int(self.upscale_factor) <= 16:
            QMessageBox.warning(self, "Configuration", "Facteur d'upscale doit être un nombre (ex: 2, 4).")
            return

        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text_edit.clear()
        self.log_messages.clear()
        
        self.log_message("Démarrage du processus complet...")

        # Créer et démarrer le worker
        worker = TaskRunner(self._run_full_pipeline_task)
        worker.signals.log.connect(self.log_message)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.finished.connect(self.on_pipeline_finished)
        worker.signals.error.connect(self.handle_error)
        self.thread_pool.start(worker)

    def _run_full_pipeline_task(self, signals: WorkerSignals):
        """
        Tâche exécutée dans un thread pour l'ensemble du pipeline.
        Utilise les signaux pour communiquer avec l'UI.
        """
        try:
            # --- 1. Cloner/Mettre à jour le dépôt Fallout 1 CE ---
            signals.log.emit("Étape 1: Clonage/Mise à jour du dépôt Fallout 1 CE...")
            signals.progress.emit(5)
            self.fallout_ce_cloned_path = self.workspace_dir / "fallout1-ce"
            try:
                if self.fallout_ce_cloned_path.exists():
                    signals.log.emit(f"Mise à jour du dépôt existant dans {self.fallout_ce_cloned_path}...")
                    repo = git.Repo(self.fallout_ce_cloned_path)
                    origin = repo.remotes.origin
                    origin.pull()
                else:
                    signals.log.emit(f"Clonage de {FALLOUT_CE_REPO_URL} vers {self.fallout_ce_cloned_path}...")
                    git.Repo.clone_from(FALLOUT_CE_REPO_URL, self.fallout_ce_cloned_path, progress=GitProgress(signals))
                signals.log.emit("Dépôt Fallout 1 CE cloné/mis à jour.")
            except Exception as e:
                signals.error.emit(f"Échec du clonage/mise à jour Git: {e}")
                return
            signals.progress.emit(10)

            # Définir les chemins des fichiers DAT
            master_dat_path = self.fallout_ce_cloned_path / "master.dat"
            critter_dat_path = self.fallout_ce_cloned_path / "critter.dat" # Ou depuis le dossier du jeu si fallout1-ce ne l'inclut pas

            if not master_dat_path.exists():
                signals.error.emit(f"master.dat non trouvé dans {self.fallout_ce_cloned_path}. Assurez-vous que le dépôt est correct ou que les fichiers du jeu sont présents.")
                # Tenter de chercher dans un sous-dossier 'game_files' ou similaire si le repo ne les contient pas directement.
                # Pour l'instant, on s'arrête ici.
                return

            # --- 2. Extraction des fichiers DAT ---
            signals.log.emit("Étape 2: Extraction des fichiers DAT (master.dat, critter.dat)...")
            extracted_assets_dir = self.workspace_dir / "extracted_assets"
            shutil.rmtree(extracted_assets_dir, ignore_errors=True) # Nettoyer les extractions précédentes
            extracted_assets_dir.mkdir(parents=True, exist_ok=True)
            
            all_extracted_frms: List[Path] = []
            
            # Traiter master.dat
            signals.log.emit("Extraction de master.dat...")
            master_dat = DatArchive(master_dat_path, signals)
            if not master_dat.load_entries(): return
            frms_master, pal_master = master_dat.extract_all_frms_and_pal(extracted_assets_dir / "master")
            all_extracted_frms.extend(frms_master)
            if pal_master and not self.color_pal_path: # Prioriser la palette de master.dat
                self.color_pal_path = pal_master
            signals.log.emit(f"{len(frms_master)} FRMs extraits de master.dat.")
            signals.progress.emit(20)

            # Traiter critter.dat (s'il existe)
            if critter_dat_path.exists():
                signals.log.emit("Extraction de critter.dat...")
                critter_dat = DatArchive(critter_dat_path, signals)
                if not critter_dat.load_entries(): return # Peut-être optionnel si critter.dat n'est pas toujours là
                frms_critter, pal_critter = critter_dat.extract_all_frms_and_pal(extracted_assets_dir / "critter")
                all_extracted_frms.extend(frms_critter)
                if pal_critter and not self.color_pal_path: # Palette de critter.dat en fallback
                    self.color_pal_path = pal_critter
                signals.log.emit(f"{len(frms_critter)} FRMs extraits de critter.dat.")
            else:
                signals.log.emit(f"critter.dat non trouvé à {critter_dat_path}, ignoré.")
            signals.progress.emit(30)

            if not self.color_pal_path or not self.color_pal_path.exists():
                # Essayer de trouver COLOR.PAL dans le dossier data du jeu cloné
                potential_pal_path = self.fallout_ce_cloned_path / "data" / "COLOR.PAL"
                if potential_pal_path.exists():
                    self.color_pal_path = potential_pal_path
                    signals.log.emit(f"Utilisation de COLOR.PAL depuis {potential_pal_path}")
                else:
                    signals.error.emit("Fichier COLOR.PAL non trouvé. Crucial pour la conversion des couleurs.")
                    return
            
            # Créer une version image de la palette pour la quantification PIL
            self.pil_palette_image_for_quantize = self.workspace_dir / "temp_palette_image.png"
            try:
                pal_data_bytes = self.color_pal_path.read_bytes()
                pil_palette_list_quant = []
                for i in range(0, 768, 3):
                    r, g, b = pal_data_bytes[i:i+3]
                    pil_palette_list_quant.extend([r * 4, g * 4, b * 4])
                while len(pil_palette_list_quant) < 768: pil_palette_list_quant.append(0)
                
                palette_img = Image.new('P', (16, 16)) # Petite image pour contenir la palette
                palette_img.putpalette(pil_palette_list_quant)
                palette_img.save(self.pil_palette_image_for_quantize)
            except Exception as e_pal_img:
                signals.error.emit(f"Impossible de créer l'image de palette temporaire: {e_pal_img}")
                return


            # --- 3. Conversion FRM -> PNG ---
            signals.log.emit("Étape 3: Conversion des fichiers FRM en PNG...")
            frm_converter = FRMConverter(signals)
            if not frm_converter.load_palette(self.color_pal_path): return
            frm_converter.set_pil_palette_image_path(self.pil_palette_image_for_quantize)


            png_output_dir = self.workspace_dir / "png_originals"
            shutil.rmtree(png_output_dir, ignore_errors=True)
            png_output_dir.mkdir(parents=True, exist_ok=True)
            
            all_png_files_grouped_by_frm: Dict[str, List[Path]] = {} # {frm_base_name: [paths_to_pngs]}
            original_frm_paths_map: Dict[str, Path] = {} # {frm_base_name: original_frm_path}


            total_frms = len(all_extracted_frms)
            for i, frm_path in enumerate(all_extracted_frms):
                # Créer un sous-dossier dans png_output_dir pour chaque FRM pour éviter les collisions de noms de frames
                relative_frm_path_from_extracted = frm_path.relative_to(extracted_assets_dir)
                # ex: master/art/critters/MYCRITTR.FRM -> master_art_critters_MYCRITTR
                # Ou plus simple, juste le nom du FRM, et on gère les collisions si nécessaire (peu probable pour les frames)
                # Pour l'upscaling en batch, un seul dossier d'entrée pour RealESRGAN est mieux.
                
                # Pour l'instant, mettons tout dans png_output_dir, les noms de frames devraient être uniques.
                pngs_for_this_frm = frm_converter.frm_to_png(frm_path, png_output_dir)
                
                # Regrouper les PNGs par nom de base du FRM original
                frm_base_name = frm_path.stem
                if frm_base_name not in all_png_files_grouped_by_frm:
                    all_png_files_grouped_by_frm[frm_base_name] = []
                all_png_files_grouped_by_frm[frm_base_name].extend(pngs_for_this_frm)
                original_frm_paths_map[frm_base_name] = frm_path


                signals.progress.emit(30 + int((i / total_frms) * 20)) # Progression de 30% à 50%
                if i % 50 == 0 : signals.log.emit(f"Converti {i}/{total_frms} FRMs en PNG...")
            signals.log.emit("Conversion FRM -> PNG terminée.")
            signals.progress.emit(50)

            # --- 4. Upscaling des PNGs ---
            signals.log.emit("Étape 4: Upscaling des images PNG avec Real-ESRGAN...")
            upscaled_png_output_dir = self.workspace_dir / "png_upscaled"
            shutil.rmtree(upscaled_png_output_dir, ignore_errors=True)
            upscaled_png_output_dir.mkdir(parents=True, exist_ok=True)

            scaler = AIScaler(self.realesrgan_exe_path, self.realesrgan_model_name, self.upscale_factor, signals)
            if not scaler.upscale_directory(png_output_dir, upscaled_png_output_dir):
                signals.error.emit("Échec de l'étape d'upscaling.")
                return
            signals.log.emit("Upscaling PNG terminé.")
            signals.progress.emit(75)

            # --- 5. Conversion PNG upscalés -> FRM ---
            signals.log.emit("Étape 5: Conversion des PNG upscalés en FRM...")
            final_frm_output_dir = self.workspace_dir / "frm_upscaled_final"
            shutil.rmtree(final_frm_output_dir, ignore_errors=True)
            final_frm_output_dir.mkdir(parents=True, exist_ok=True)

            # Regrouper les PNGs upscalés par nom de base du FRM original
            upscaled_png_files_grouped_by_frm: Dict[str, List[Path]] = {}
            for upscaled_png_path in sorted(upscaled_png_output_dir.glob("*.png")):
                # Le nom du fichier PNG upscalé est le même que l'original (ex: MYFRM_d0_f0.png)
                original_png_name_parts = upscaled_png_path.stem.split('_d')
                frm_base_name_of_upscaled = original_png_name_parts[0]
                
                if frm_base_name_of_upscaled not in upscaled_png_files_grouped_by_frm:
                    upscaled_png_files_grouped_by_frm[frm_base_name_of_upscaled] = []
                upscaled_png_files_grouped_by_frm[frm_base_name_of_upscaled].append(upscaled_png_path)

            total_original_frms_to_repack = len(original_frm_paths_map)
            repacked_count = 0
            for frm_base_name, original_frm_path_val in original_frm_paths_map.items():
                if frm_base_name in upscaled_png_files_grouped_by_frm:
                    # Déterminer le chemin de sortie relatif pour le nouveau FRM
                    # ex: si original_frm_path_val était extracted_assets/master/art/critters/X.FRM
                    # alors le nouveau FRM doit aller dans final_frm_output_dir/master/art/critters/X.FRM
                    relative_path_structure = original_frm_path_val.relative_to(extracted_assets_dir)
                    output_subdir_for_frm = final_frm_output_dir / relative_path_structure.parent
                    output_subdir_for_frm.mkdir(parents=True, exist_ok=True)
                    
                    frm_converter.png_to_frm(upscaled_png_files_grouped_by_frm, output_subdir_for_frm, original_frm_path_val)
                else:
                    signals.log.emit(f"Avertissement: Aucun PNG upscalé trouvé pour {frm_base_name}, FRM original non recréé.")
                
                repacked_count += 1
                signals.progress.emit(75 + int((repacked_count / total_original_frms_to_repack) * 25)) # Progression de 75% à 100%
                if repacked_count % 20 == 0: signals.log.emit(f"Reconverti {repacked_count}/{total_original_frms_to_repack} FRMs...")

            signals.log.emit("Conversion PNG upscalés -> FRM terminée.")
            signals.log.emit(f"Les fichiers FRM upscalés sont dans: {final_frm_output_dir}")
            signals.log.emit("Vous pouvez maintenant copier le contenu de ce dossier dans votre installation de Fallout 1 CE (faites une sauvegarde avant !).")
            signals.progress.emit(100)

        except Exception as e:
            signals.error.emit(f"Une erreur majeure est survenue dans le pipeline: {e}\n{traceback.format_exc()}")
        finally:
            # Nettoyer l'image de palette temporaire
            if self.pil_palette_image_for_quantize and self.pil_palette_image_for_quantize.exists():
                try:
                    self.pil_palette_image_for_quantize.unlink()
                except Exception:
                    pass # Ignorer les erreurs de suppression


# --- Classe utilitaire pour la progression Git ---
import traceback

class GitProgress(git.remote.RemoteProgress):
    """Affiche la progression des opérations Git via les signaux PyQt."""
    def __init__(self, signals: WorkerSignals):
        super().__init__()
        self.signals = signals
        self.last_percentage = -1

    def update(self, op_code, cur_count, max_count=None, message=''):
        if max_count:
            percentage = int((cur_count / max_count) * 100)
            if percentage != self.last_percentage: # Éviter de spammer les logs/progress
                # La progression Git est souvent en plusieurs étapes (counting, compressing, receiving)
                # On peut essayer de la mapper sur une portion de la barre de progression globale
                # Par exemple, si le clonage est 10% du total, et qu'on est à 50% du clonage -> 5% global.
                # Pour l'instant, on logge juste.
                self.signals.log.emit(f"Git: {git.remote.RemoteProgress.OP_CODE_MAP.get(op_code, op_code)} - {cur_count}/{max_count} {message}")
                self.last_percentage = percentage
        else:
            self.signals.log.emit(f"Git: {git.remote.RemoteProgress.OP_CODE_MAP.get(op_code, op_code)} - {cur_count} {message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Appliquer un style sombre simple (optionnel)
    # app.setStyle("Fusion")
    # dark_palette = QPalette()
    # dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    # dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    # # ... (définir d'autres couleurs pour les boutons, les champs, etc.)
    # app.setPalette(dark_palette)

    main_window = FalloutUpscalerApp()
    main_window.show()
    sys.exit(app.exec())

