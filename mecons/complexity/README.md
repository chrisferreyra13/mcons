# Complexity metrics
## Lempel-Ziv complexity (LZc)
It derives from the lack of compressibility of the input data matrix, taking into account patterns of activity in space and time. Strictly only reflects differentiation (and not integration)[1]. The normalised LZc is computed by dividing the raw value by the value obtained for the same binary input sequence randomly shuffled [1][2].

Reading recommendations:
* [1]: Schartner et al., 2015.
* [2]: Schartner et al., 2017.

## Amplitude coalition entropy (ACE)
It reflects the entropy over time of the constitution (coalition) of the set of most active channels [2]. It is similar to Lempel-Ziv complexity in the sense that it quantifies variability in space and time of the activity [1]. The normalised value is computed in the same way as LZc [1][2].

Reading recommendations:
* [1]: Schartner et al., 2015.
* [2]: Schartner et al., 2017.

## Synchrony coalition entropy (SCE)
It reflects the entropy over time of the constitution (coalition) of the set of synchronous channels [2]. It quantifies variability in the relationship between pairs of channels [1].

Reading recommendations:
* [1]: Schartner et al., 2015.
* [2]: Schartner et al., 2017.

## Perturbational complexity index (PCI)
[TO-ADD]  
Complete brief description.

Reading recommendations:
* [3]: Casali et al., 2013.

## Permutation entropy (PE)
[TO-ADD]  
Complete brief description.

## References
1. Schartner, M., Seth, A., Noirhomme, Q., Boly, M., Bruno, M. A., Laureys, S., & Barrett, A. (2015). Complexity of multi-dimensional spontaneous EEG decreases during propofol induced general anaesthesia. PloS one, 10(8), e0133532.
2. Schartner, M. M., Pigorini, A., Gibbs, S. A., Arnulfo, G., Sarasso, S., Barnett, L., ... & Barrett, A. B. (2017). Global and local complexity of intracranial EEG decreases during NREM sleep. Neuroscience of consciousness, 2017(1), niw022.
3. Casali, A. G., Gosseries, O., Rosanova, M., Boly, M., Sarasso, S., Casali, K. R., ... & Massimini, M. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. Science translational medicine, 5(198), 198ra105-198ra105.
