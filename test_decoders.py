#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:43:41 2025

@author: gaston
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions as F
from sionna.phy.fec.polar import PolarSCLDecoder, PolarSCDecoder
from sionna.phy.fec.linear import OSDecoder
from sionna.phy.fec.ldpc.decoding import LDPCBPDecoder
# from sionna.phy.fec.polar.decoding import PolarBPDecoder
from time import time
from tensorflow.keras.models import load_model
import tensorflow as tf

# Check if GPU is available and being used
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("No GPU available, running on CPU only\n")
else:
    print("GPU OK\n")

time_init = time()
max_time = 24*3600
m_size, l_width, font_size = 3.5, 1, 8

# %% CHOOSE SETTINGS FOR SIMULATIION - everything you need to change is here

# Modulation parameters
# 1/2/3/4/... - QAM/PSK - bin/gray/SP
MOD_TYPE, MOD_ORDER, SymOrder = 'QAM', 4, 'SP'
N_turbo_iter = 10  # For turbo-demodulation - choose 1 for no turbo-demod

# Coding and decoding parameters
TYPE, n, k = 'POLAR', 64, 32
DECOD_METHOD = 'NN'  # SC/SCL/OSD/soft-OSD/BP/FGBP/NN
# cn_type='boxplus','boxplus-phi','minsum'
OSD_order, list_size, BP_iter = 1, 16, 50
INTERLEAVER, SCRAMBLER, CPU_ONLY, PROGRESS = True, False, False, True
NNDecoder_path = 'Simulations/POLAR-n64-k32_SBND-RecECCT-denoiser_512bs_5-5dB_16QAM_Hstandright_gen-random_gray_BICM ULTIMO'

# Simulation parameters
EbN0dB_list = np.arange(5, 6.1, 0.5)
min_block_errors = 100
nb_sim_max = 500
nb_sim_min = 2
test_batch_size = 2**9

# %% Construct modulator
const = F.get_modulator(MOD_TYPE, MOD_ORDER, SymOrder)
nt = int(n/const.num_bits_symbol)
matrix_c = np.array([list(format(i, f'0{MOD_ORDER}b')) for i in range(2**MOD_ORDER)], dtype=int).T
demod_prior = F.Demod_prior(matrix_c, const._constellation)

# %% Save things
save_path = f'Simulations/{TYPE}-n{n}-k{k}_{2**MOD_ORDER}{MOD_TYPE}_{DECOD_METHOD}'
if 'OSD' in DECOD_METHOD:
    save_path = save_path + f'{OSD_order}'
elif DECOD_METHOD == "SCL":
    save_path = save_path + f'{list_size}'
elif "BP" in DECOD_METHOD:
    save_path = save_path + f'{BP_iter}'

save_path = save_path + f'_{SymOrder}' + '_BICM'*INTERLEAVER + '/'

# %% Matrix loading

if DECOD_METHOD == 'NN':
    H = np.loadtxt(NNDecoder_path + f"/H_{TYPE}_n{n}_k{k}.txt", dtype=int)
    G = np.loadtxt(NNDecoder_path + f"/G_{TYPE}_n{n}_k{k}.txt", dtype=int)
    A = np.loadtxt(NNDecoder_path + f"/A_{TYPE}_n{n}_k{k}.txt", dtype=int)
else:
    H = np.loadtxt(f"PC-matrices/{TYPE}_N{n}_K{k}.txt", dtype=int)

    if 'SC' in DECOD_METHOD:
        frozen_idx = np.argwhere(F.H_2_frozenlookup(H)).T[0]
        G, A = F.H_to_G_A(H)
    else:
        G, H, sys_idx = F.get_systematic_code(H, PCM_FORM='normal')
        A = np.eye(n)[:, sys_idx]

if np.sum(G@H.T % 2) != 0:
    raise ValueError(
        'PC matrix criterion not satisfied -> Invalid (G,H) pair!')

# %% Load decoder

if DECOD_METHOD == 'SC':
    dec = PolarSCDecoder(frozen_idx, n)
elif DECOD_METHOD == 'SCL':
    dec = PolarSCLDecoder(frozen_idx, n, list_size=list_size)
elif DECOD_METHOD == 'OSD':
    dec = OSDecoder(enc_mat=G, t=OSD_order)
    # dec = F.OSD_symbol(OSD_order, G, H, const)
elif DECOD_METHOD == 'soft-OSD':
    dec = F.OSD_soft(OSD_order, G, H, const)
elif DECOD_METHOD == 'BP':
    dec = LDPCBPDecoder(pcm=H, trainable=False, cn_update='boxplus-phi',
                        num_iter=BP_iter, hard_out=False, llr_max=30)
    # dec = F.BP(H, BP_iter)
elif DECOD_METHOD == 'FGBP':
    dec = F.FGBP(n, frozen_idx, BP_iter)
    # dec = PolarBPDecoder(frozen_pos=frozen_idx, n=n, hard_out=False, num_iter=BP_iter)
elif DECOD_METHOD == 'NN':
    dec = load_model(f'{NNDecoder_path}/decod_model.h5', compile=False,
                     custom_objects={'llr_to_abs_syn': F.llr_to_abs_syn,
                                     'EmbeddingLayer': F.EmbeddingLayer,
                                     'RecurrentEncoder': F.RecurrentEncoder,
                                     'FinalLayer': F.FinalLayer,
                                     'llr_to_RNN': F.llr_to_RNN,
                                     'EncoderLayer': F.EncoderLayer,
                                     'Encoder': F.Encoder,
                                     'myMultiHeadAttention': F.myMultiHeadAttention,
                                     'GRU': F.CustomGRU})
    # dec = tf.keras.layers.TFSMLayer(f'{NNDecoder_path}/decod_model', call_endpoint='serving_default')
    save_path = NNDecoder_path + \
        f'/test-{SymOrder}' + '-scrambled'*SCRAMBLER + '/'
        
#%%

# mask = tf.cast(F.create_mask(H), tf.float32)

# inp = tf.keras.Input(shape=(n,), name='Input')
# x = dec.get_layer('Preprocessing')(inp)
# x = dec.get_layer('EmbeddingLayer')(x)
# rec = dec.get_layer('RecEncoder')
# for _ in range(5):
#     x = rec(x, mask=mask)
# out = dec.get_layer('DecodingLayer')(x)
# dec2 = tf.keras.Model(inp, out)

# for i in range(len(dec.layers)):
#     print(dec.layers[i].weights == dec2.layers[i].weights)


# %% Save file and simulation information

# save file copy
os.makedirs(save_path)
shutil.copy(__file__, save_path+'script.py')
shutil.copy('functions.py', save_path+'functions.py')

# save matrices
np.savetxt(save_path+'G_{}_n{}_k{}.txt'.format(TYPE, n, k), G, fmt='%d')
np.savetxt(save_path+'H_{}_n{}_k{}.txt'.format(TYPE, n, k), H, fmt='%d')


# %% BER and FER study

# Compute ber and fer of uncoded

print('Computing BER for uncoded')
ber_uncoded, fer_uncoded = [], []
kt = int(k/const.num_bits_symbol)

for i_ebn0 in range(len(EbN0dB_list)):
    print('Eb/N0 = {}dB'.format(EbN0dB_list[i_ebn0]))
    sigma2 = 1/(10**(EbN0dB_list[i_ebn0]/10)*const.num_bits_symbol)
    nb_bin_errors, nb_block_errors = 0, 0

    i_sim = 0
    while (nb_block_errors < min_block_errors*20) & (i_sim < nb_sim_max*10):
        i_sim += 1

        # message generation and coding
        messages = np.random.randint(0, 2, size=(test_batch_size, n))

        # modulation
        mod_messages = F.modulate(const, messages)

        # channel
        noise = np.sqrt(sigma2/2) * (np.random.randn(test_batch_size,nt) + 1j*np.random.randn(test_batch_size, nt))
        x_noise = mod_messages + noise

        # hard demodulation
        messages_rec = (F.demodulate_soft(const, x_noise, sigma2) > 0)*1

        # add up biinary and frame errors
        nb_bin_errors += np.sum((messages != messages_rec)*1)
        nb_block_errors += np.sum(np.sum((messages != messages_rec)*1, axis=1) != 0)

    ber_uncoded.append(nb_bin_errors/(i_sim*k*test_batch_size))
    fer_uncoded.append(nb_block_errors/(i_sim*test_batch_size))


# Compute BER and FER for SC decoder
print(f'\nComputing BER and FER for {DECOD_METHOD}')

BER = np.zeros((N_turbo_iter, len(EbN0dB_list)))
BER_cw = np.zeros((N_turbo_iter, len(EbN0dB_list)))
FER = np.zeros((N_turbo_iter, len(EbN0dB_list)))
FER_cw = np.zeros((N_turbo_iter, len(EbN0dB_list)))
FER_MLB = np.zeros(len(EbN0dB_list))

# number of total frames sent
errors_per_position = np.zeros((k), dtype=int)
nb_frames = 0

frame_errors = 0
for i_ebn0 in range(len(EbN0dB_list)):
    sigma2 = 1/(10**(EbN0dB_list[i_ebn0]/10)*(k/n)*const.num_bits_symbol)

    # if i_ebn0 > 0 and frame_errors < 20:
    #     break

    i_sim = 0
    while ((np.min(FER[:, i_ebn0]) < min_block_errors) | (i_sim < nb_sim_min)) & (i_sim < nb_sim_max) & (time()-time_init < max_time):
        i_sim += 1
        u_k = np.random.randint(0, 2, size=(test_batch_size, k))
        c_n = (u_k@G) % 2
        c_scrambled, mask = F.scramble(c_n, bypass=not SCRAMBLER)

        # interleaver
        idx_inter, idx_deinter = F.interleaver(c_scrambled.shape, multiblock=True, bypass=not INTERLEAVER)
        c_inter = F.reorder(c_scrambled, idx_inter)

        # modulation
        mod_codewords = F.modulate(const, c_inter)

        # channel
        noise = np.sqrt(sigma2/2) * (np.random.randn(*mod_codewords.shape) + 1j*np.random.randn(*mod_codewords.shape))
        x_noise = mod_codewords + noise

        # Iterative decoding
        llr_m_a = np.zeros(c_n.shape)
        
        # initialize for best iteration
        syn_0_best = 0

        for i in range(N_turbo_iter):

            # Demodulator posterior
            if i == 0:
                llr_m_p = F.demodulate_soft(const, x_noise, sigma2)*-1.0
            else:
                llr_m_p = demod_prior(x_noise.flatten(), llr_m_a.flatten(), sigma2, decision_method='llr').reshape(-1, n)

            # Extrinsic transfer to decoder
            llr_m_e = llr_m_p - llr_m_a
            llr_d_a = F.reorder(llr_m_e, idx_deinter)

            # initialize decoding values
            if i == 0:
                llr_d_p = llr_d_a

            # check errored codewords
            # idx_wrong = np.sum((llr_d_p<0)@H.T %2, axis=1) != 0

            # Descrambling (does nothing if SCRAMBLE=False):
            llr_in = llr_d_a*(mask*-2+1)

            # DECODING

            if 'SC' in DECOD_METHOD:
                messages_rec = dec(llr_in*-1.0).numpy().astype(int)*-1.0
                codewords_rec = (messages_rec@G) % 2
                llr_d_p = codewords_rec*-2.0 + 1.0
            elif DECOD_METHOD == 'OSD':
                llr_d_p = dec(llr_in*-1.0).numpy()*-2.0 + 1.0
            elif DECOD_METHOD == 'soft-OSD':
                _, llr_d_p = dec(x_noise, llr_in, sigma2, idx_inter)
            elif DECOD_METHOD == 'BP':
                llr_d_p = dec(llr_in*-1.0).numpy()*-1.0
                # llr_d_p = dec(llr_d_a[idx_wrong]*-1.0)
            elif DECOD_METHOD == 'FGBP':
                _, llr_d_p = dec(llr_in)
                # llr_d_p = dec(llr_d_a*-1.0).numpy()
            elif DECOD_METHOD == 'NN':
                noise_est = dec(llr_in*sigma2*-1.0).numpy()*-1.0
                if dec.layers[-1].output.shape[-1] == n:
                    llr_d_p = np.sign(llr_in)*noise_est
                    # llr_d_p = np.clip((np.sign(llr_in)*noise_est), -15, 15)                    
                else:
                    u_ref_sign = ((llr_in < 0)@A % 2)*-2.0 + 1
                    aux = u_ref_sign*noise_est
                    llr_d_p = F.boxplus_matmul(aux, G)

            # Re-scrambling (again, does nothing if SCRAMBLE=False)
            llr_d_p = llr_d_p*(mask*-2+1)

            # obtain hard outputs
            c_n_hat_scrambled = (llr_d_p < 0)*1
            c_n_hat = F.descramble(c_n_hat_scrambled, mask)
            u_k_hat = (c_n_hat@A % 2).astype(int)

            # Extrinsic transfer to demapper
            llr_d_e = llr_d_p - llr_d_a
            llr_m_a = F.reorder(llr_d_e, idx_inter)
            
            # now the "tricks" for NN decoding: early stopping and best iteration
            if (DECOD_METHOD == 'NN') and (N_turbo_iter>1):
                
                # compute the number of null-syndromes
                syn_0 = np.sum(np.sum(c_n_hat@H.T % 2, axis=1) == 0)
            
                # stop if all parity checks are satisfied and copy errors for next iterations
                # if syn_0==test_batch_size:
                #     BER[i:N_turbo_iter, i_ebn0] += np.sum(np.bitwise_xor(u_k, u_k_hat))
                #     BER_cw[i:N_turbo_iter, i_ebn0] += np.sum(np.bitwise_xor(c_n, c_n_hat))
                #     FER[i:N_turbo_iter, i_ebn0] += np.sum(np.sum(np.bitwise_xor(u_k, u_k_hat), axis=1) != 0)
                #     FER_cw[i:N_turbo_iter, i_ebn0] += np.sum(np.sum(np.bitwise_xor(c_n, c_n_hat), axis=1) != 0)
                #     break
            
                # "best iteration" trick: keep iterations with largest number of null syndromes
                if syn_0 > syn_0_best:
                    syn_0_best = syn_0
                    llr_d_p_best = llr_d_p
            
                # if last iteration, then replace by the best iteration
                if (i == N_turbo_iter-1) and (DECOD_METHOD=='NN'):
                    c_n_hat_scrambled = (llr_d_p_best < 0)*1
                    c_n_hat = F.descramble(c_n_hat_scrambled, mask)
                    u_k_hat = (c_n_hat@A % 2).astype(int)

            # Add up errors
            BER[i, i_ebn0] += np.sum(np.bitwise_xor(u_k, u_k_hat))
            BER_cw[i, i_ebn0] += np.sum(np.bitwise_xor(c_n, c_n_hat))
            FER[i, i_ebn0] += np.sum(np.sum(np.bitwise_xor(u_k,u_k_hat), axis=1) != 0)
            FER_cw[i,i_ebn0] += np.sum(np.sum(np.bitwise_xor(c_n, c_n_hat), axis=1) != 0)

            # if (i == N_turbo_iter-1) and (not INTERLEAVER):
            #     # Maximum likelihood bound: c_n_hat_itrlv = u_k_hat -> encoding -> interleaving
            #     aux = u_k_hat@G %2
            #     c_n_hat_itrlv = F.reorder(aux, idx_inter)
            #     FER_MLB[i_ebn0] += F.MLbound(const, x_noise, c_n_hat_itrlv, c_inter)

        #########################################################

        # Show progress bar in console
        F.progress(EbN0dB_list[i_ebn0], np.min(FER[:, i_ebn0]).astype(int), min_block_errors, i_sim, nb_sim_min)

    frame_errors = np.min(FER[:, i_ebn0])
    BER[:, i_ebn0] = BER[:, i_ebn0]/(i_sim*k*test_batch_size)
    BER_cw[:, i_ebn0] = BER_cw[:, i_ebn0]/(i_sim*n*test_batch_size)
    FER[:, i_ebn0] = FER[:, i_ebn0]/(i_sim*test_batch_size)
    FER_cw[:, i_ebn0] = FER_cw[:, i_ebn0]/(i_sim*test_batch_size)
    FER_MLB[i_ebn0] = FER_MLB[i_ebn0]/(i_sim*test_batch_size)

    nb_frames += i_sim*test_batch_size
    print('#Sims = {} | BER = {:.2e} | FER = {:.2e}'.format(i_sim, BER[-1, i_ebn0], FER[-1, i_ebn0]))
    # print('#ML bound | FER = {:.2e}'.format( FER_MLB[i_ebn0]))

    # Print error rates into txt file
    with open(save_path+'error_rates.txt', 'w') as f:
        l = EbN0dB_list.shape[0]
        f.write('FER\n')
        for j in range(N_turbo_iter):
            for i in range(l):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_list[i], FER[j, i]))
            f.write('\n')

        f.write('\nBER\n')
        for j in range(N_turbo_iter):
            for i in range(l):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_list[i], BER[j, i]))
            f.write('\n')

        f.write('\nBER_cw\n')
        for j in range(N_turbo_iter):
            for i in range(l):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_list[i], BER_cw[j, i]))
            f.write('\n')

        f.write('\nFER_cw\n')
        for j in range(N_turbo_iter):
            for i in range(l):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_list[i], FER_cw[j, i]))

# %% ########################## ERROR STUDIES FOR DECODER ##################################################
    m_size = 3.5
    l_width = 1
    font_size = 7

    plt.figure()
    plt.semilogy(EbN0dB_list, ber_uncoded[0:len(
        EbN0dB_list)], '--k', label='uncoded', markersize=m_size, linewidth=l_width)
    for i in range(N_turbo_iter):
        plt.semilogy(EbN0dB_list, BER[i, :], '.-', label=f'Iter {i+1}', markersize=m_size, linewidth=l_width)
    for i in range(N_turbo_iter):
        plt.semilogy(EbN0dB_list, BER_cw[i, :], '--g', markersize=m_size, linewidth=l_width)
    plt.title('BER {} ({},{}) - {} - {}{}'.format(TYPE,n, k, DECOD_METHOD, 2**MOD_ORDER, MOD_TYPE))
    plt.xlabel('Eb/N0 (dB)')
    plt.legend(prop={'size': font_size})
    plt.grid("on", which='both', ls='--')
    plt.savefig(save_path+'BER.eps', format='eps', dpi=600)
    plt.close()

    plt.figure()
    plt.semilogy(EbN0dB_list, fer_uncoded[0:len(
        EbN0dB_list)], '--k', label='uncoded', markersize=m_size, linewidth=l_width)
    for i in range(N_turbo_iter):
        plt.semilogy(EbN0dB_list, FER[i, :], '.-', label=f'Iter {i+1}', markersize=m_size, linewidth=l_width)
    # for i in range(N_turbo_iter):
        # plt.semilogy(EbN0dB_list, FER_cw[i, :], '--g', markersize=m_size, linewidth=l_width)
    if not INTERLEAVER:
        plt.semilogy(EbN0dB_list, FER_MLB, '-or', label='ML bound',markersize=m_size, linewidth=l_width)
    plt.title('FER {} ({},{}) - {} - {}{}'.format(TYPE,n, k, DECOD_METHOD, 2**MOD_ORDER, MOD_TYPE))
    plt.xlabel('Eb/N0 (dB)')
    plt.legend(prop={'size': font_size})
    plt.grid("on", which='both', ls='--')
    plt.savefig(save_path+'FER.eps', format='eps', dpi=600)
    plt.show()
