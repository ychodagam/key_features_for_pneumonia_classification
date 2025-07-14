
"""
Module: features
Provides functions for extracting radiomic wavelet-based features, including additional GLCM metrics and image-based descriptors.
"""

import numpy as np
import pywt
from skimage.feature import graycomatrix
from skimage.filters import frangi
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from scipy.ndimage import sobel

def build_modified_coif1(perturbation=0.05):
    coif1 = pywt.Wavelet('coif1')
    #dec_lo = np.array(coif1.dec_lo, dtype=float)
    #dec_lo_modified = dec_lo.copy()
    #dec_lo_modified[3] *= (1 + perturbation)
    #dec_lo_modified /= dec_lo_modified.sum()
    #N = len(dec_lo_modified)
    #dec_hi = [((-1)**(k+1)) * dec_lo_modified[N-1-k] for k in range(N)]
    #rec_lo = dec_lo_modified[::-1].tolist()
    #rec_hi = dec_hi[::-1]
    return coif1

def discretize_fixed_bin(image, binWidth=25):
    image_scaled = image * 255.0
    return np.floor(image_scaled / binWidth).astype(np.uint8)

def compute_glcm_features(image, levels=None, epsilon=1e-8):
    if levels is None:
        levels = int(np.max(image)) + 1

    glcm = graycomatrix(image,
                        distances=[1],
                        angles=[0],
                        levels=levels,
                        symmetric=True,
                        normed=True)
    p = glcm[:, :, 0, 0]
    i_vals = np.arange(levels)
    j_vals = np.arange(levels)

    mu_x = np.sum(i_vals * np.sum(p, axis=1))
    mu_y = np.sum(j_vals * np.sum(p, axis=0))

    cluster_tendency = 0
    for i in range(levels):
        for j in range(levels):
            cluster_tendency += ((i + j) - (mu_x + mu_y))**2 * p[i, j]

    p_sum = np.zeros(2 * levels - 1)
    for i in range(levels):
        for j in range(levels):
            p_sum[i + j] += p[i, j]
    sum_entropy = -np.sum(p_sum * np.log2(p_sum + epsilon))

    return cluster_tendency, sum_entropy

def compute_glrlm(image, levels=None):
    if levels is None:
        levels = int(np.max(image)) + 1
    rows, cols = image.shape
    glrlm = {}
    max_run = 0
    for r in range(rows):
        c = 0
        while c < cols:
            current = image[r, c]
            run_length = 1
            c += 1
            while c < cols and image[r, c] == current:
                run_length += 1
                c += 1
            if current not in glrlm:
                glrlm[current] = {}
            glrlm[current][run_length] = glrlm[current].get(run_length, 0) + 1
            max_run = max(max_run, run_length)

    R = np.zeros((levels, max_run))
    for i in range(levels):
        if i in glrlm:
            for run_length, count in glrlm[i].items():
                R[i, run_length - 1] = count
    return R

def compute_glrlm_features(image, levels=None, epsilon=1e-8):
    R = compute_glrlm(image, levels)
    total_runs = np.sum(R)
    if total_runs == 0:
        return 0, 0, 0
    p_rl = R / total_runs

    run_entropy = -np.sum(p_rl * np.log2(p_rl + epsilon))

    sre = 0
    for j in range(R.shape[1]):
        sre += np.sum(p_rl[:, j]) / ((j + 1)**2)

    col_sum = np.sum(R, axis=0)
    rln = np.sum(col_sum**2)
    rln_norm = rln / (total_runs**2)

    return run_entropy, sre, rln_norm

def _compute_additional_glcm_metrics(image, levels=None, epsilon=1e-8):
    if levels is None:
        levels = int(np.max(image)) + 1

    glcm = graycomatrix(image,
                        distances=[1],
                        angles=[0],
                        levels=levels,
                        symmetric=True,
                        normed=True)
    p = glcm[:, :, 0, 0]
    i_vals = np.arange(levels)[:, None]
    j_vals = np.arange(levels)[None, :]

    dist = np.abs(i_vals - j_vals)
    Id   = np.sum(p / (1.0 + dist))
    Idn  = np.sum(p / (1.0 + dist / levels))
    Idm  = np.sum(p / ((1.0 + dist)**2))
    Idmn = np.sum(p / (1.0 + (dist**2) / (levels**2)))

    px = p.sum(axis=1)
    py = p.sum(axis=0)
    HXY  = -np.sum(p    * np.log2(p    + epsilon))
    HX   = -np.sum(px   * np.log2(px   + epsilon))
    HY   = -np.sum(py   * np.log2(py   + epsilon))
    HXY1 = -np.sum(p    * np.log2((px[:,None]*py[None,:]) + epsilon))
    HXY2 = -np.sum((px[:,None]*py[None,:]) * np.log2((px[:,None]*py[None,:]) + epsilon))
    Imc1 = (HXY - HXY1) / max(HX, HY)
    Imc2 = np.sqrt(max(0.0, 1.0 - np.exp(-2.0*(HXY2 - HXY))))

    return Id, Idn, Idm, Idmn, Imc1, Imc2

def compute_wavelet_features(image, wavelet, quant_binWidth=25):
    coeffs = pywt.dwt2(image, wavelet)
    LL, (LH, HL, HH) = coeffs

    features = {}

    for band, name in zip([LH, HH], ['LH', 'HH']):
        features[f'{name}_mean']   = np.mean(band)
        features[f'{name}_std']    = np.std(band)
        features[f'{name}_energy'] = np.sum(band**2)

    LH_disc = discretize_fixed_bin(LH, binWidth=quant_binWidth)
    HH_disc = discretize_fixed_bin(HH, binWidth=quant_binWidth)

    ct, se = compute_glcm_features(LH_disc, levels=int(np.max(LH_disc))+1)
    features['LH_glcm_ClusterTendency'] = ct
    features['LH_glcm_SumEntropy']      = se

    Id, Idn, Idm, Idmn, Imc1, Imc2 = _compute_additional_glcm_metrics(
        LH_disc, levels=int(np.max(LH_disc))+1
    )
    features['LH_glcm_Id']    = Id
    features['LH_glcm_Idn']   = Idn
    features['LH_glcm_Idmn']  = Idmn
    features['LH_glcm_Idm']   = Idm
    features['LH_glcm_Imc1']  = Imc1
    features['LH_glcm_Imc2']  = Imc2

    Id, Idn, Idm, Idmn, Imc1, Imc2 = _compute_additional_glcm_metrics(
        HH_disc, levels=int(np.max(HH_disc))+1
    )
    features['HH_glcm_Id']    = Id
    features['HH_glcm_Idn']   = Idn
    features['HH_glcm_Idmn']  = Idmn
    features['HH_glcm_Idm']   = Idm
    features['HH_glcm_Imc1']  = Imc1
    features['HH_glcm_Imc2']  = Imc2

    run_entropy_LH, sre_LH, rln_norm_LH = compute_glrlm_features(LH_disc,
                                                                 levels=int(np.max(LH_disc))+1)
    features['LH_glrlm_RunEntropy']                     = run_entropy_LH
    features['LH_glrlm_ShortRunEmphasis']               = sre_LH
    features['LH_glrlm_RunLengthNonUniformityNormalized'] = rln_norm_LH

    run_entropy_HH, sre_HH, rln_norm_HH = compute_glrlm_features(HH_disc,
                                                                 levels=int(np.max(HH_disc))+1)
    features['HH_glrlm_RunEntropy']                     = run_entropy_HH
    features['HH_glrlm_RunLengthNonUniformityNormalized'] = rln_norm_HH

    # --- Append new image-based texture features ---
    texture_features = extract_texture_features(image)
    features.update(texture_features)

    return features

def extract_texture_features(image):
    entropy_map = entropy(img_as_ubyte(image), disk(5))
    edge_map = np.hypot(sobel(image, axis=0), sobel(image, axis=1))
    frangi_map = frangi(image)
    return {
        'grad_entropy_mean': np.mean(entropy_map),
        'sobel_edge_mean': np.mean(edge_map),
        'frangi_mean': np.mean(frangi_map)
    }
