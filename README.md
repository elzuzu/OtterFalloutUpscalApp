**Comment utiliser l'application :**

1.  **Installation des dépendances Python :**
    * Assurez-vous d'avoir Python installé.
    * Installez toutes les bibliothèques nécessaires avec :
        ```bash
        pip install -r requirements.txt
        ```

2.  **Télécharger Real-ESRGAN :**
    * L'application vérifie automatiquement la présence de l'exécutable et du modèle. S'ils sont absents, ils seront téléchargés dans le répertoire du projet.
    * Vous pouvez tout de même déclencher manuellement le téléchargement avec :
        ```bash
        python main.py --download-real-esrgan chemin/vers/dossier
        ```
      (par défaut, le téléchargement se fait dans le dossier courant.)
    * Alternativement, rendez-vous sur la page des releases de [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases) et téléchargez la version `realesrgan-ncnn-vulkan`. Extrayez `realesrgan-ncnn-vulkan.exe` ainsi que le dossier `models`.

3.  **Lancer l'application :**
    * Exécutez simplement :
        ```bash
        python main.py
        ```

4.  **Configuration dans l'interface :**
    * **Répertoire de travail :** Choisissez un dossier où l'application stockera le dépôt cloné, les fichiers extraits, les PNG, etc.
    * **Real-ESRGAN (.exe) :** Indiquez le chemin complet vers `realesrgan-ncnn-vulkan.exe`.
    * **Modèle Real-ESRGAN (nom) :** Entrez le nom du modèle que vous souhaitez utiliser (par exemple, `realesrgan-x4plus-anime` ou `RealESRGAN_x4plus`). Ce nom doit correspondre aux fichiers `.param` et `.bin` présents à côté de l'exécutable ou dans son sous-dossier `models`.
    * **Facteur d'Upscale :** Généralement `4` pour un upscale x4.

5.  **Démarrer le processus :**
    * Cliquez sur "Démarrer le Processus Complet".
    * Suivez les logs et la barre de progression.
    * À la fin de l'upscaling, une fenêtre de comparaison avant/après vous permettra de valider le résultat.

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
* **Détection automatique du type d'asset :** Les fichiers FRM sont désormais catégorisés (personnages ou textures) selon leur chemin. Des modèles Real-ESRGAN différents sont appliqués automatiquement pour chaque catégorie.

## Nouvelles fonctionnalités

* **Détection améliorée des assets** grâce à l'analyse d'image et à de nouveaux motifs de chemin.
* **Pipeline `HybridUpscaler`** combinant Upscayl, ComfyUI et Real‑ESRGAN selon le type d'asset avec validation de qualité (SSIM/PSNR).
* **Profils configurables** : les paramètres des moteurs et leurs seuils peuvent être sauvegardés dans `~/.ofua_profiles`. Des profils `speed`, `quality`, `balanced` et `fallout_optimized` sont inclus.
* **Script `setup_enhanced.py`** pour préparer automatiquement l'environnement (installation des dépendances et vérifications système).

Pour exécuter ce script :

```bash
python setup_enhanced.py
```

N'hésitez pas à créer vos propres profils pour adapter vitesse et qualité. Pour toute question, ouvrez un ticket sur le dépôt.


## Guide de dépannage

Si l'application ne se lance pas ou plante en cours d'exécution :

1. Vérifiez que toutes les dépendances Python sont installées (`pip install -r requirements.txt`).
2. Assurez-vous que `realesrgan-ncnn-vulkan.exe` est présent et exécutable.
3. Consultez le fichier `upscaler.log` dans votre répertoire de travail pour plus de détails.
