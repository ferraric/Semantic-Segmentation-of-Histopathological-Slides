# Semantic Segmentation of Histopathological Slides

## Project Description
Mycosis fungoides (MF) is a slowly progressing but potentially life-threatening skin disease. If appropriate treatment is applied in early stages, the patients encounter a normal survival rate. Eczema on the other hand is a very common benign inflammatory skin disease whose clinical and histopathological features are similar to early stage MF. The early distinction of MF and eczema is of utmost importance in order to get the appropriate treatment for the patient.

The diagnosis of MF and eczema is usually based on the microscopic examination of histopathological slides (thin slices of stained tissue) and relies mostly on the presence of atypical lymphocytes in the epidermis. However, histological distinction from eczema may be very difficult due to similar histological features. Thus the final diagnosis of early MF is very challenging and relies on guidelines that include multiple  eighted criteria such as the clinical appearance, histopathology, immune-histology, and molecular pathology.

 
 

Goal

The aim of this project is to develop machine learning techniques that can detect and highlight relevant features in histopathological images in order to support clinicians in distinguishing MF from eczema. In particular, one goal is to develop methods for semantic segmentation of such images that can automatically recognize spongiotic areas in the epidermis as well as regions in which lymphocytes are present. Building upon that, another goal is to develop techniques for quantifying the presence of spongiotic areas and lymphocytes in the epidermis. Ultimately, these techniques should become part of a system that can assist clinicians in distinguishing MF from eczema.

The project will be based on a a dataset consisting of 201 Hematoxylin-Eosin (HE) stained whole slide images (WSI) scanned at 20x magnification and including the label of the diagnosis at the WSI-level (MF or eczema) as well as other types of annotations.