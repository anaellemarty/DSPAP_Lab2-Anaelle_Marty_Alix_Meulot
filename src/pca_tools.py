
# Fonction pour afficher le cercle de corrélation de la PCA

def plot_correlation_circle(pca, features, axis1=1, axis2=2):
    """
    Affiche le cercle de corrélation pour deux axes principaux d'une PCA.

    Paramètres
    ----------
    pca : sklearn.decomposition.PCA
        Objet PCA déjà ajusté (fit) sur les données.
    features : list of str
        Liste des noms des variables d'origine (features).
    axis1 : int
        Numéro du premier axe principal (1-indexé, ex: 1 pour PC1).
    axis2 : int
        Numéro du second axe principal (1-indexé, ex: 2 pour PC2).

    Retourne
    --------
    None. Affiche le cercle de corrélation.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    loadings = pca.components_.T
    eigvals = pca.explained_variance_
    # Indices pour axes (0-indexé)
    i, j = axis1-1, axis2-1
    coords = loadings[:, [i, j]] * np.sqrt(eigvals[[i, j]])
    fig, ax = plt.subplots(figsize=(6,6))
    circ = plt.Circle((0,0), 1.0, fill=False, linewidth=1.0, color='grey', linestyle='--')
    ax.add_artist(circ)
    ax.axhline(0, linewidth=0.5, color='grey')
    ax.axvline(0, linewidth=0.5, color='grey')
    for (x, y), name in zip(coords, features):
        ax.arrow(0, 0, x, y, head_width=0.03, head_length=0.04, length_includes_head=True, linewidth=0.8, color='C0')
        ax.text(x*1.08, y*1.08, name, fontsize=9)
    ax.set_xlabel(f"PC{axis1}")
    ax.set_ylabel(f"PC{axis2}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Cercle de corrélation PCA (PC{axis1}–PC{axis2})")
    plt.tight_layout()
    plt.show()
    plt.close('all')