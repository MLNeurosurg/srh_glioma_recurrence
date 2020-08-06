# RecurSRH
Code repository for paper titled "Rapid, label-free detection of diffuse glioma recurrence using intraoperative stimulated Raman histology and deep neural networks" published in Neuro-Oncology, July 2020

# Background
Detection of glioma recurrence remains a challenge in modern neuro-oncology. Noninvasive radiographic imaging is unable to definitively differentiate true recurrence versus pseudoprogression. Even in biopsied tissue, it can be challenging to differentiate recurrent tumor and treatment effect. We hypothesized that intraoperative stimulated Raman histology (SRH) and deep neural networks can be used to improve the intraoperative detection of glioma recurrence. 
# Methods
We used fiber-laser-based SRH, a label-free, non-consumptive, high-resolution microscopy method (<60 secs per 1 x 1 mm2) to image a cohort of patients (n = 35) with suspected recurrent gliomas who underwent biopsy or resection. The SRH images were then used to train a convolutional neural network (CNN) and develop an inference algorithm to detect viable recurrent glioma. Following network training, the performance of the CNN was tested for diagnostic accuracy in a retrospective cohort (n = 48).
# Results 
Using patch-level CNN predictions, the inference algorithm returned a single Bernoulli distribution for the probability of tumor recurrence for each surgical specimen or patient. The external SRH validation dataset consisted of 48 patients (recurrent, 30; pseudoprogression, 18), and we achieved a diagnostic accuracy of 95.8%.
# Conclusion
SRH with CNN-based diagnosis can be used to improve the intraoperative detection of glioma recurrence in near-real time. Our results provide insight into how optical imaging and computer vision can be combined to augment conventional diagnostic methods and improve the quality of specimen sampling at glioma recurrence.
