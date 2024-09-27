# Syndrome-Based Neural Decoder
Implementation of model-free syndrome-based neural decoders.

## Disclaimer
The repository is still in progress. It may contain errors, so the interested user may have to get their hands a little bit dirty.

## Short summary
This repository contains a collection of works from different articles, some of them still unpublished (see Reference). Feel free to play around with the training and simulation parameters to either reproduce the results in the papers or produce your own. Some instructions are provided in SBND.py

## Packages needed
- Tensorflow
- matplotlib
- numpy
- commpy (for higher-order modulations)
- itertools (for sparsifying the parity-check matrix)
- maybe others...

## To run a simulation...
1) Download everything to a folder
2) Open the SBND.py file
3) Choose the training and simulation parameters
4) Run

## Reference
My IEEE profile: https://ieeexplore.ieee.org/author/706500273613698

"Scalable Syndrome-based Neural Decoders for Bit-Interleaved Coded Modulations" (2024)

@INPROCEEDINGS{10625146,
  author={De Boni Rovella, Gastón and Benammar, Meryem and Benaddi, Tarik and Meric, Hugo},
  booktitle={2024 IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN)}, 
  title={Scalable Syndrome-based Neural Decoders for Bit-Interleaved Coded Modulations}, 
  year={2024},
  volume={},
  number={},
  pages={341-346},
  keywords={Recurrent neural networks;Bit error rate;Modulation;Computer architecture;Interleaved codes;Transformers;Decoding},
  doi={10.1109/ICMLCN59089.2024.10625146}}

"Improved Syndrome-based Neural Decoder for Linear Block Codes" (2023)

@INPROCEEDINGS{10436980,
  author={De Boni Rovella, Gastón and Benammar, Meryem},
  booktitle={GLOBECOM 2023 - 2023 IEEE Global Communications Conference}, 
  title={Improved Syndrome-based Neural Decoder for Linear Block Codes}, 
  year={2023},
  volume={},
  number={},
  pages={5689-5694},
  keywords={Training;Codes;Scalability;Bit error rate;Machine learning;Decoding;Signal to noise ratio},
  doi={10.1109/GLOBECOM54140.2023.10436980}}
    
## License
This repo is MIT licensed.
