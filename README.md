**Comment utiliser l'application :**

1.  **Installation des dépendances Python :**
    * Assurez-vous d'avoir Python installé.
    * Installez toutes les bibliothèques nécessaires avec :
        ```bash
        pip install -r requirements.txt
        ```

2.  **Télécharger Real-ESRGAN :**
    * Vous pouvez récupérer automatiquement l'exécutable avec :
        ```bash
        python OFUA.py --download-real-esrgan chemin/vers/dossier
        ```
      (par défaut, le téléchargement se fait dans le dossier courant.)
    * Alternativement, rendez-vous sur la page des releases de [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases) et téléchargez la version `realesrgan-ncnn-vulkan`. Extrayez `realesrgan-ncnn-vulkan.exe` ainsi que le dossier `models`.

3.  **Lancer l'application :**
    * Exécutez simplement :
        ```bash
        python OFUA.py
        ```

4.  **Configuration dans l'interface :**
    * **Répertoire de travail :** Choisissez un dossier où l'application stockera le dépôt cloné, les fichiers extraits, les PNG, etc.
    * **Real-ESRGAN (.exe) :** Indiquez le chemin complet vers `realesrgan-ncnn-vulkan.exe`.
    * **Modèle Real-ESRGAN (nom) :** Entrez le nom du modèle que vous souhaitez utiliser (par exemple, `realesrgan-x4plus-anime` ou `RealESRGAN_x4plus`). Ce nom doit correspondre aux fichiers `.param` et `.bin` présents à côté de l'exécutable ou dans son sous-dossier `models`.
    * **Facteur d'Upscale :** Généralement `4` pour un upscale x4.

5.  **Démarrer le processus :**
    * Cliquez sur "Démarrer le Processus Complet".
    * Suivez les logs et la barre de progression.

6.  **Résultats :**
    * Une fois terminé, les fichiers `.FRM` upscalés se trouveront dans le sous-dossier `frm_upscaled_final` de votre répertoire de travail.
    * La structure des dossiers à l'intérieur de `frm_upscaled_final` (par exemple, `master/art/critters/`) devrait correspondre à celle attendue par le jeu.
    * **TRÈS IMPORTANT :** Avant de remplacer des fichiers dans votre installation de Fallout 1 CE, faites une **SAUVEGARDE COMPLÈTE** de votre jeu. Copiez ensuite le contenu de `frm_upscaled_final` dans le répertoire racine de Fallout 1 CE, en écrasant les fichiers existants si nécessaire (ou en les plaçant dans un dossier `data` ou `mods` si le moteur le supporte pour les overrides).

**Limitations et améliorations possibles :**

* **Gestion des erreurs :** Bien qu'il y ait une gestion basique, des erreurs spécifiques aux formats de fichiers ou à Real-ESRGAN peuvent nécessiter un débogage plus approfondi.
* **Repackaging des .DAT :** Actuellement, l'application produit des fichiers `.FRM` individuels. Elle ne recrée pas les fichiers `master.dat` ou `critter.dat`. Pour une intégration "propre", il faudrait idéalement repacker ces archives, ce qui est plus complexe et risqué. Cependant, de nombreux moteurs de jeu modernes (comme potentiellement Fallout 1 CE) peuvent charger des assets depuis des dossiers en vrac, ce qui rend le repacking moins critique.
* **Choix du modèle Real-ESRGAN :** L'efficacité de l'upscale dépendra grandement du modèle choisi. Expérimentez avec des modèles spécifiquement entraînés pour le pixel art ou les jeux rétro.
* **Performance :** Le processus peut être long, surtout l'étape d'upscaling.
* **Interface utilisateur :** L'interface est fonctionnelle mais pourrait être améliorée (plus d'options, meilleure présentation des erreurs, etc.).
* **Offsets des sprites :** Lors de la reconversion PNG vers FRM, les offsets (`offset_x`, `offset_y`) des frames sont actuellement mis à 0. Pour un rendu parfait, il faudrait idéalement les recalculer ou les ajuster en fonction de la nouvelle taille des sprites, ou reprendre ceux des FRM originaux si l'upscale ne déplace pas le "centre" du sprite.
* **Palette pour la quantification :** La méthode `Image.quantize(palette=Image.open(self.pil_palette_image_path))` est utilisée. La qualité de la quantification est cruciale. `dither=Image.Dither.FLOYDSTEINBERG` aide.

J'espère que cette application vous sera un excellent point de départ ! N'hésitez pas si vous avez des questions ou si vous rencontrez des problèmes.Okay, I've generated a comprehensive Python application with a PyQt6 GUI that aims to automate the Fallout 1 CE asset upscaling process as you described.

The application includes:
1.  A user interface for configuration and monitoring.
2.  Git cloning/updating for the Fallout 1 CE repository.
3.  Extraction of `master.dat` and `critter.dat` files.
4.  Conversion of Fallout's `.FRM` image files to `.PNG`.
5.  A mechanism to call an external Real-ESRGAN executable for the actual AI upscaling.
6.  Conversion of the upscaled `.PNG` images back to `.FRM`, attempting to use the game's original color palette.
7.  Saving of the new `.FRM` files into an organized output directory.
8.  Logging and progress bar feedback.

You'll find the Python code in the immersive document block.

**Before you run it, please note these important points:**

* **Real-ESRGAN is External:** The Python script *does not* include Real-ESRGAN itself. You need to:
    * Download the `realesrgan-ncnn-vulkan.exe` (for Windows) and corresponding model files (`.param` and `.bin`) from the [official Real-ESRGAN GitHub releases](https://github.com/xinntao/Real-ESRGAN/releases) or another trusted source.
    * A model specifically trained or suited for pixel art/retro games is highly recommended for best results (e.g., something other than the generic photo or anime models, though `realesrgan-x4plus` or `realesrgan-x4plus-anime` can be starting points).
* **Configuration is Key:** You *must* configure the paths to your Real-ESRGAN executable and the name of the model you wish to use within the application's UI. These settings are saved for future sessions.
* **`COLOR.PAL` is Crucial:** The application attempts to extract `COLOR.PAL` from `master.dat` or find it in the cloned game's `data` directory. This palette file is essential for correct color handling during the FRM-PNG-FRM conversions.
* **Backup Your Game:** **Always, always, always back up your Fallout 1 CE game installation before replacing any game files with upscaled assets.**
* **Output:** The application will output the upscaled `.FRM` files into a structured directory (e.g., `workspace/frm_upscaled_final/master/art/critters/`). You would then manually copy these into your Fallout 1 CE game directory, ideally into a location the engine checks for overrides (like a `data` folder or a specific `mods` folder if supported by Fallout 1 CE). The application does *not* repack the `.DAT` files.
* **Dependencies:** You'll need to install `PyQt6`, `Pillow`, `numpy`, and `GitPython`. You can do this via pip:
    ```bash
    pip install PyQt6 Pillow numpy GitPython
    ```

I've tried to make the code as robust as possible, including error handling and user feedback. The process of extracting proprietary game formats, converting, upscaling, and reconverting can be complex, so there might be edge cases or specific files that present challenges.

Let me know if you have any questions as you get it set up and runni
