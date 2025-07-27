Functions and Formulations
==========================

Dataset preprocessing
----------------------

Assume an MSI dataset represented as a 3D array of dimensions :math:`H \times W \times N`, where :math:`H` and :math:`W` are the spatial dimensions of the tissue sample, and :math:`N` is the number of measured ions (*m/z*). Each ion image :math:`I_k \in \mathbb{R}^{H \times W}` captures the spatial intensity distribution of the :math:`k^{\text{th}}` *m/z* value across the sample, for :math:`k = 1, \dots, N`. Conversely, each pixel location :math:`(i, j)`, with :math:`i = 1, \dots, H` and :math:`j = 1, \dots, W`, corresponds to a spectrum :math:`S_{i,j} \in \mathbb{R}^N`, which contains the intensity values for all :math:`N` *m/z* channels at that location.

Normalization
^^^^^^^^^^^^^

MassVision currently supports six spectrum normalization methods as follows:

*TIC normalization:*  
Total Ion Current normalization scales each spectrum by the sum of its intensity values. For each pixel, the normalization is defined as:

.. math::

   \text{TIC}(S_{i,j}) = \frac{S_{i,j}}{\sum_{k=1}^{N} S_{i,j}[k]}

Here, :math:`S_{i,j}[k]` denotes the intensity of the :math:`k^{\text{th}}` ion at pixel :math:`(i, j)`, and the denominator represents the total ion current at that location. This operation ensures that all normalized spectra have unit total intensity, helping to mitigate the effects of acquisition-related fluctuations and tissue heterogeneity.

*TSC normalization:*  
Total Signal Current normalization scales each spectrum by the sum of intensity values that exceed a user-defined threshold. For each pixel, the normalization is defined as:

.. math::

   \text{TSC}(S_{i,j}; \tau) = \frac{S_{i,j}}{\sum_{k=1}^{N} S_{i,j}[k] \cdot \mathbf{1}_{\{S_{i,j}[k] > \tau\}}}

Here, :math:`\tau` is the intensity threshold, and :math:`\mathbf{1}_{\{S_{i,j}[k] > \tau\}}` is the indicator function, which equals 1 when :math:`S_{i,j}[k] > \tau` and 0 otherwise. This normalization includes only signal components above the threshold in the total, reducing the influence of background noise and low-intensity fluctuations while preserving biologically relevant variation.

*Reference normalization:*  
In reference normalization, each spectrum is scaled by the intensity of a specific reference ion. For each pixel, the normalization is defined as:

.. math::

   \text{Ref}(S_{i,j}; k^*) = \frac{S_{i,j}}{S_{i,j}[k^*]}

Here, :math:`k^* \in \{1, \dots, N\}` is the index of the chosen reference ion (corresponding to a particular *m/z* value), and :math:`S_{i,j}[k^*]` is its intensity at pixel :math:`(i,j)`. This normalization preserves relative ion abundances while anchoring the scale to a biologically or experimentally relevant signal. The reference ion should be consistently present across spectra and stable in intensity to ensure reliable scaling.

*Statistical scaling:*  
This is a family of normalization methods in which each spectrum is scaled by a scalar summary statistic of its intensity values. For each pixel, the normalized spectrum is defined as:

.. math::

   \text{SCALE}_f(S_{i,j}) = \frac{S_{i,j}}{f(S_{i,j})}

Available options in MassVision for :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}_{>0}` include:

* *Mean normalization:* :math:`f(S_{i,j}) = \frac{1}{N} \sum_{k=1}^{N} S_{i,j}[k]`

- *Median normalization:* :math:`f(S_{i,j}) = \operatorname{median}(S_{i,j})`

- *RMS normalization:*  :math:`f(S_{i,j}) = \sqrt{\frac{1}{N} \sum_{k=1}^{N} S_{i,j}[k]^2}`

This formulation supports flexible normalization strategies: for example, median normalization offers robustness to outliers and noise, while RMS normalization emphasizes higher-intensity signals and better reflects spectral energy.

Pixel aggregation
^^^^^^^^^^^^^^^^^

Spatial denoising can be achieved by applying a local aggregation operation over a sliding kernel across each ion image. A square kernel of side length :math:`w`, with a symmetric stride (pitch) :math:`s` in both spatial directions, is applied independently to each ion image. For all integer indices :math:`i` and :math:`j`, the stride-aligned kernel center is given by:

.. math::

   (x', y') = (i \cdot s,\, j \cdot s),

and the output value at that location is computed as:

.. math::

   \hat{I}_k(x', y') = \underset{(u,v) \in \mathcal{N}_w(x', y')}{\text{AGG}} \, I_k(u, v)


where :math:`\mathcal{N}_w(x', y')` denotes the :math:`w \times w` neighborhood centered at :math:`(x', y')`. The aggregation operator :math:`\text{AGG}` can be instantiated by a user-defined function *f* such as *min*, *max*, *sum*, or *mean*, each computing a scalar summary over the specified neighborhood.



