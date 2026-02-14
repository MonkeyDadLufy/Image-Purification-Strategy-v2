import pydicom
import numpy as np
from skimage.transform import radon, iradon

import os
import pydicom
import numpy as np
import SimpleITK as sitk
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import sys
from utils_functions import calculate_snr_cnr_phan, affine_from_dicom, col_mas_av


def apply_ld_sinogram_full(high_path, mas_des, print_logs=False, mod=False, gaussian_noise = False):
    """
    Apply low-dose simulation to a high-dose CT slice.
    
    Parameters:
    high_path : str
        Path to the high-dose DICOM file.
    mas_des : float
        Desired mAs for low-dose simulation.
    print_logs : bool, optional
    mod : bool, optional
    gaussian_noise : bool, optional
    
    If mod=True → use mAs correction with pitch (phan version).
    If print_logs=True → print intermediate steps.
    If gaussian_noise=False → apply only Poisson noise.
    """

    def log(msg):
        if print_logs:
            print(msg)

    #=========== CONSTANTES SIMULACIÓN BAJA DOSIS ==============
    a = 0.015
    b = 0.1002

    dcm = pydicom.dcmread(high_path)
    exp_time_elem = dcm.get((0x0018, 0x1150))  # ExposureTime
    current_elem  = dcm.get((0x0018, 0x1151))  # XRayTubeCurrent

    exp_time = float(exp_time_elem.value) if exp_time_elem else 0
    current  = float(current_elem.value) if current_elem else 0

    # ==== CÁLCULO DEL EXPOSURE mAs =====
    if exp_time_elem is None or current_elem is None:
        exposure_mAs = 200
        log("ExposureTime/XRayTubeCurrent missing → Using default 200 mAs")
    else:
        if mod:
            pitch = 0.7
            exposure_mAs = ((exp_time / 1000) * current) / pitch
            log(f"Using mod=True (phan) → exposure_mAs = {exposure_mAs}")
        else:
            exposure_mAs = (exp_time / 1000) * current
            log(f"Using mod=False → exposure_mAs = {exposure_mAs}")

    flux_per_mAs = 1000
    k = a * mas_des + b
    inc_flux = exposure_mAs * flux_per_mAs

    CONT_FIG = 500

    #=========== LECTURA SLICE ALTA DOSIS ============
    h_slice = dcm.pixel_array.astype(np.float64)
    h_slice = (h_slice - np.min(h_slice)) / (np.max(h_slice) - np.min(h_slice))
    log("High-dose slice loaded and normalized.")

    #=========== SINOGRAMA ALTA DOSIS ============
    theta = np.linspace(0.0, 180.0, max(h_slice.shape), endpoint=False)
    sinogram = radon(h_slice, theta=theta) * (1.0 / CONT_FIG)
    log("Radon transform (high dose sinogram) computed.")

    #=========== TRANSMISIÓN ============
    t_nd = np.exp(-sinogram)

    #=========== FLUJO TRANSMITIDO SIMULADO ============
    transm_flux = k * inc_flux * t_nd
    log("Transmitted flux for low dose computed.")

    #=========== RUIDO =============
    if gaussian_noise:
        # Sin ruido gaussiano
        transm_noisy = np.random.poisson(transm_flux)
        log("Poisson noise applied (no Gaussian).")
    else:
        # Con ruido Poisson + Gaussiano
        transm_noisy = np.random.poisson(transm_flux) + np.random.normal(0, np.sqrt(transm_flux)*0.05)
        log("Poisson + Gaussian noise applied.")

    #=========== SINOGRAMA BAJA DOSIS ============
    sino_ld = -np.log(transm_noisy / (k * inc_flux))
    sino_ld[transm_noisy <= 0] = 0
    sinogram_ld = sino_ld * CONT_FIG
    log("Low-dose sinogram computed.")

    #=========== RECONSTRUCCIÓN ============
    low_slice = iradon(sinogram_ld, theta=theta, filter_name='hann')
    log("CT low-dose slice reconstructed (FBP).")

    return h_slice, low_slice, sinogram_ld


def apply_ld_sinogram_from_png(png_path, output_png_path, mas_des=50.0, print_logs=False):
    """
    Apply low-dose simulation to a PNG image directly (no DICOM).
    
    Parameters:
    -----------
    png_path : str
        Path to the input high-dose PNG image.
    output_png_path : str
        Path where to save the output low-dose PNG image.
    mas_des : float, optional
        Desired mAs for low-dose simulation. Default is 50.0.
    print_logs : bool, optional
        If True, print intermediate steps. Default is False.
    
    Returns:
    --------
    h_slice : ndarray
        Normalized high-dose image (float64, range [0, 1]).
    low_slice : ndarray
        Reconstructed low-dose image.
    sinogram_ld : ndarray
        Low-dose sinogram.
    """
    def log(msg):
        if print_logs:
            print(msg)

    #=========== CONSTANTES SIMULACIÓN BAJA DOSIS ==============
    a = 0.015
    b = 0.1002

    #=========== LECTURA PNG ALTA DOSIS ============
    from PIL import Image
    img = Image.open(png_path).convert('L')  # Grayscale
    h_slice = np.array(img, dtype=np.float64)
    h_slice = (h_slice - np.min(h_slice)) / (np.max(h_slice) - np.min(h_slice))
    log(f"High-dose PNG loaded from {png_path} and normalized.")

    #=========== SIMULACIÓN DE EXPOSICIÓN BAJA DOSIS ============
    # Sin DICOM, asumimos una exposición de referencia predeterminada (200 mAs)
    exposure_mAs = 200.0
    log(f"Reference exposure_mAs = {exposure_mAs}")

    flux_per_mAs = 1000
    k = a * mas_des + b
    inc_flux = exposure_mAs * flux_per_mAs

    CONT_FIG = 500

    #=========== SINOGRAMA ALTA DOSIS ============
    theta = np.linspace(0.0, 180.0, max(h_slice.shape), endpoint=False)
    sinogram = radon(h_slice, theta=theta) * (1.0 / CONT_FIG)
    log("Radon transform (high dose sinogram) computed.")

    #=========== TRANSMISIÓN ============
    t_nd = np.exp(-sinogram)

    #=========== FLUJO TRANSMITIDO SIMULADO ============
    transm_flux = k * inc_flux * t_nd
    log("Transmitted flux for low dose computed.")

    #=========== RUIDO (Poisson + Gaussian) =============
    transm_noisy = np.random.poisson(transm_flux) + np.random.normal(0, np.sqrt(transm_flux) * 0.05)
    log("Poisson + Gaussian noise applied.")

    #=========== SINOGRAMA BAJA DOSIS ============
    sino_ld = -np.log(transm_noisy / (k * inc_flux))
    sino_ld[transm_noisy <= 0] = 0
    sinogram_ld = sino_ld * CONT_FIG
    log("Low-dose sinogram computed.")

    #=========== RECONSTRUCCIÓN ============
    low_slice = iradon(sinogram_ld, theta=theta, filter_name='hann')
    log("CT low-dose slice reconstructed (FBP).")

    #=========== GUARDADO PNG BAJA DOSIS ============
    # Normalizar la imagen reconstruida a [0, 255] para PNG
    low_slice_uint8 = np.clip((low_slice - np.min(low_slice)) / (np.max(low_slice) - np.min(low_slice)) * 255, 0, 255).astype(np.uint8)
    output_img = Image.fromarray(low_slice_uint8, mode='L')
    output_img.save(output_png_path)
    log(f"Low-dose PNG saved to {output_png_path}")

    return h_slice, low_slice, sinogram_ld

def simulation_ldct_colorectal(colorectal_path, base_dose_path, ld_method="1", save_path=False):
    patients_with_mas, list_patients_with_mas = col_mas_av(colorectal_path)



def metrics_simulation_phan(phan_path, list_phantoms, list_mas, base_dose_path, ld_method="1", save_path=False):
    '''

    Function to directly simulate low dose in phantoms and calculate SNR, CNR, and sigma metrics.

    param 
        phan_path: Path to the folder containing the phantoms.
        list_phantoms: List of phantom names to process.
        list_mas: List of mAs values (same length as number of phantoms high dose).
        base_dose_path: Folder path with high dose images.
        ld_method: Method for low dose sinogram simulation (default is "1").
        save_path: If str (path), saves simulated slices and metrics CSV.
    '''
    prom_snr_sim1, prom_snr_sim2, prom_cnr_sim, prom_sigma_sim = [], [], [], []
    prom_snr_real1, prom_snr_real2, prom_cnr_real, prom_sigma_real = [], [], [], []

    low_slices_arrays = []
    low_slices_arrays_real = []

    results = []  # para acumular resultados de cada phantom

    a = 0
    for i in range(len(list_phantoms)):
        print(f"Processing phantom: {list_phantoms[i]}")
        if "high" not in list_phantoms[i]:
            continue
        phantom = list_phantoms[i]
        images = os.path.join(phan_path, phantom)
        affine_matrix = affine_from_dicom(os.path.join(phan_path, phantom))
        list_dcms = sorted(os.listdir(images))

        # High dose
        list_dcms_h = sorted(os.listdir(base_dose_path))

        # --- Real LD ---
        snrs_real1, snrs_real2, cnrs_real, sigmas_real, list_slices_real = [], [], [], [], []
        for j in range(len(list_dcms)):
            dcm = os.path.join(phan_path, phantom, list_dcms[j])
            slice_dcm = pydicom.dcmread(dcm).pixel_array
            snr_real1, snr_real2, cnr_real, sigma_real = calculate_snr_cnr_phan(slice_dcm)
            snrs_real1.append(snr_real1)
            snrs_real2.append(snr_real2)
            cnrs_real.append(cnr_real)
            sigmas_real.append(sigma_real)
            list_slices_real.append(slice_dcm.T)
        low_slices_arrays_real.append(list_slices_real)

        # --- Simulated LD ---
        snrs_sim1, snrs_sim2, cnrs_sim, sigmas_sim, list_slices_sim = [], [], [], [], []
        mas = list_mas[a] #// 2
        a += 1

        for j in range(len(list_dcms_h)):
            dcm_h = os.path.join(base_dose_path, list_dcms_h[j])
            h_slice, low_slice, low_sinogram = apply_ld_sinogram_full(dcm_h, mas)
            snr_sim1, snr_sim2, cnr_sim, sigma_sim = calculate_snr_cnr_phan(low_slice)
            snrs_sim1.append(snr_sim1)
            snrs_sim2.append(snr_sim2)
            cnrs_sim.append(cnr_sim)
            sigmas_sim.append(sigma_sim)
            list_slices_sim.append(low_slice.T)

        if save_path:
            save_path_ld = os.path.join(save_path, f'low_slices_sim_{phantom}_mas_{mas}.nii')
            save_path_real = os.path.join(save_path, f'low_slices_real_{phantom}_mas_{mas}.nii')
            #nib.save(nib.Nifti1Image(np.array(list_slices_sim), affine_matrix), save_path_ld)
            #nib.save(nib.Nifti1Image(np.array(list_slices_real), affine_matrix), save_path_real)

            img_array_3D = np.stack(list_slices_real[::-1], axis=-1)
            ld_array_3D = np.stack(list_slices_sim[::-1], axis=-1)
            img_nii = nib.Nifti1Image(img_array_3D, affine_matrix)  # Save aSxis for data (just identity)
            ld_nii = nib.Nifti1Image(ld_array_3D, affine_matrix)
            img_nii.to_filename(save_path_real)  # Save as NiBabel file
            ld_nii.to_filename(save_path_ld)  # Save as NiBabel file
            #np.save(os.path.join(save_path, f'low_slices_sim_{phantom}_mas_{mas}.npy'), np.array(list_slices_sim))

        # Guardamos promedios
        mean_snr_sim1, mean_snr_sim2, mean_cnr_sim, mean_sigma_sim = np.mean(snrs_sim1), np.mean(snrs_sim2), np.mean(cnrs_sim), np.mean(sigmas_sim)
        mean_snr_real1, mean_snr_real2, mean_cnr_real, mean_sigma_real = np.mean(snrs_real1), np.mean(snrs_real2), np.mean(cnrs_real), np.mean(sigmas_real)

        prom_snr_sim1.append(mean_snr_sim1)
        prom_snr_sim2.append(mean_snr_sim2)
        prom_cnr_sim.append(mean_cnr_sim)
        prom_sigma_sim.append(mean_sigma_sim)
        low_slices_arrays.append(list_slices_sim)

        prom_snr_real1.append(mean_snr_real1)
        prom_snr_real2.append(mean_snr_real2)
        prom_cnr_real.append(mean_cnr_real)
        prom_sigma_real.append(mean_sigma_real)

        print("Number of slices real:", len(snrs_real1  ))
        print("Number of slices simulated:", len(snrs_sim1))

        print(
            f'phantom: {phantom}, mAs: {mas}, '
            f'snr 1: {mean_snr_sim1:.4f} ± {np.std(snrs_sim1):.4f}, '
            f'snr real1: {mean_snr_real1:.4f} ± {np.std(snrs_real1):.4f}, '
            f'snr 2: {mean_snr_sim2:.4f} ± {np.std(snrs_sim2):.4f}, '
            f'snr real2: {mean_snr_real2:.4f} ± {np.std(snrs_real2):.4f}, '
            f'cnr: {mean_cnr_sim:.4f} ± {np.std(cnrs_sim):.4f}, '
            f'cnr real: {mean_cnr_real:.4f} ± {np.std(cnrs_real):.4f}, '
            f'sigma: {mean_sigma_sim:.4f} ± {np.std(sigmas_sim):.4f}, '
            f'sigma real: {mean_sigma_real:.4f} ± {np.std(sigmas_real):.4f}'
        )

        # Agregamos al diccionario de resultados
        results.append({
            "phantom": phantom,
            "mAs": mas,
            "snr_sim1_mean": mean_snr_sim1,
            "snr_sim2_mean": mean_snr_sim2,
            "snr_real1_mean": mean_snr_real1,
            "snr_real2_mean": mean_snr_real2,
            "cnr_sim_mean": mean_cnr_sim,
            "cnr_real_mean": mean_cnr_real,
            "sigma_sim_mean": mean_sigma_sim,
            "sigma_real_mean": mean_sigma_real,
        })

    # Guardamos CSV
    if save_path:
        df = pd.DataFrame(results)
        csv_path = os.path.join(save_path, "metrics_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Resultados guardados en {csv_path}")

    return prom_snr_sim1, prom_snr_sim2, prom_cnr_sim, prom_sigma_sim, prom_snr_real1, prom_snr_real2, prom_cnr_real, prom_sigma_real, low_slices_arrays, low_slices_arrays_real