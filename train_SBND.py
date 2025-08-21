#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:39:05 2024

@author: g.de-boni-rovella
"""

# Imports

import shutil, os, sys, time
# os.environ["TF_USE_LEGACY_KERAS"] = "1"
# import tf_keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions as F
# tf.keras.backend.set_floatx('float64') # uncomment for further floating point precision

time_init = time.time()
max_time = 48  # maximum simulation time in hours
plt.rcParams['lines.linewidth'], plt.rcParams['lines.markersize'] = 1, 5 # for prettier figures

'''
This script was made by Gastón De Boni Rovella in order for the reader to
reproduce (or explore) the results of the two following articles:
    - "Improved Syndrome-based Neural Decoder for Linear Block Codes"
    - "Syndrome-Based Neural Decoding for Higher-Order Modulations"
    - "Optimizing the Parity-Check Matrix for Syndrome-Based Neural Decoders",
along with other works associated with this application (see IEEExplore).

A few comments are given below.

1) PCM_FORM is for the parity-check (PC) matrix structure:
    - 'normal' leaves the PC matrix as is
    - 'standright' standardizes the matrix from right to left
    - 'standleft' idem, from left to right
    - 'sparsifyk' applies a k-th order sparsifying algorithm
    - 'random' applies random linear combinations
Feel free to try several options and analyze the resulting matrix.

2) bit_gen is for the bit generation for training:
    - 'zeros' transmits the all-zero codeword
    - 'ones' transmits the codeword that corresponds to the all-one message
    - 'random' transmits randomly generated codewords in every epoch
    - '01' tarnsmits a codeword with similar number of 0s and 1s
    
Contact me (gdeboni@fing.edu.uy) if you have questions or corrections.
'''

print("TensorFlow version:", tf.__version__)
if not tf.config.list_physical_devices('GPU'):
    print("No GPU available, running on CPU only")
else:
    print("GPU OK")

# %% CHOOSE SIMULATION SETTINGS IN THIS SECTION

# Modulator - example for 16-QAM: MOD_ORDER, MOD_TYPE = 4, 'QAM'
MOD_ORDER, MOD_TYPE, SymOrder = 4, 'QAM', 'gray'  # 1/2/3/4/... - QAM/PSK and bin/gray/SP

# Error correcting code
TYPE, n, k = 'POLAR', 128, 64  # 'BCH' 'POLAR'
PCM_FORM = 'standright' # 'normal', 'standright', 'standleft', 'sparsify1', 'sparsify2', 'random'
bit_gen = 'random'  # 'zeros', 'ones', 'random', '01'
INTERLEAVER = True  # True or False for the presence of a bit-interleaver

# Neural decoder
type_output = 'decoder' # select 'decoder' for message-oriented and 'denoiser' for codeword-oriented.
type_NN = 'RNN'  # 'MLP', 'RNN', 'ECCT', 'RecECCT', 'DoubleECCT'

# Training parameters
EbN0dB_train = [5, 7]  # min and max Eb/N0 used in training
batch_size = 2**12
epochs, patience = 10000, 1000
time_lim_hours = 24
learning_rate = 1e-3

# Decoder testing parameters
EbN0dB_test = np.arange(0,9.5,1)
min_block_errors = 500
nb_sim_max = 100000
test_batch_size = 2**9

# Other parameters
VERBOSE, PROGRESS = 0, True # select "0, False" for no output, or "1, True" for all outputs

# %% Get modulator and code
const = F.get_modulator(MOD_TYPE, MOD_ORDER, mode=SymOrder)
nt = int(n/const.num_bits_symbol)

# get parity-check matrix
H = np.loadtxt('PC-matrices/{}_N{}_K{}.txt'.format(TYPE, n, k)).astype(int)

# get generator and PC matrix for the equivalent systematic code
if TYPE == '--POLAR': # always systematic (no reason to choose non-systematic)
    G, A = F.H_to_G_A(H)
    H = F.change_H(H, PCM_FORM)
else:
    G, H, sys_idx = F.get_systematic_code(H, PCM_FORM)
    A = np.eye(n)[:, sys_idx]

# %% Create save folder and copy script

add_to_filename = f'{2**MOD_ORDER}{MOD_TYPE}_H{PCM_FORM}_gen-{bit_gen}_{SymOrder}' +'_BICM'*INTERLEAVER
save_path = f'Simulations/{TYPE}-n{n}-k{k}_SBND-{type_NN}-{type_output}_{batch_size}bs_{EbN0dB_train[0]}-{EbN0dB_train[1]}dB_{add_to_filename}/'

os.makedirs(save_path)
shutil.copy(__file__, save_path+'script.py')
shutil.copy('functions.py', save_path+'functions.py')
np.savetxt(save_path+'/G_{}_n{}_k{}.txt'.format(TYPE, n, k), G, fmt='%d')
np.savetxt(save_path+'/H_{}_n{}_k{}.txt'.format(TYPE, n, k), H, fmt='%d')
np.savetxt(save_path+'/A_{}_n{}_k{}.txt'.format(TYPE, n, k), A, fmt='%d')

# %% """"""""""
###########################################################################################
########################## Definition of the models  ######################################
###########################################################################################

# NN DECODER
if type_NN == 'MLP':
    size_factor, depth = 7, 10  # if it is RNN
    decod_model = F.get_MLP_decoder(type_output, H, size_factor, activation='linear', depth=depth)
if type_NN == 'RNN':
    RNN_steps, RNN_size_factor, depth = 5, 5, 5  # if it is RNN
    decod_model = F.get_RNN_decoder(type_output, H, RNN_steps, RNN_size_factor, activation='linear', depth=depth)
elif type_NN == 'ECCT':
    d_model, nb_head, nb_encoder = 128, 8, 10  # if it is transformer
    decod_model = F.get_ECCT(type_output, d_model,nb_encoder, nb_head, H, mask=True)
elif type_NN == 'RecECCT':
    d_model, nb_head, steps = 128, 8, 10  # if it is transformer
    decod_model = F.get_recurrent_ECCT(type_output, d_model, steps, nb_head, H, mask=True)
elif type_NN == 'DoubleECCT':
    d_model, nb_head, nb_encoder = 128, 8, 10  # if it is transformer
    decod_model = F.get_double_ECCT(type_output, d_model, nb_encoder, nb_head, H, F.change_H(H, 'standleft'))
else:
    raise ValueError('Invalid NN architecture!')

loss_fcn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, name='Adam_decod')
decod_model.compile(optimizer=optimizer, loss=loss_fcn, metrics=[F.sign_error_logits, F.fer_block_logits])
with open(save_path+'/modelsummary.txt', 'w') as f:
    decod_model.summary(print_fn=lambda x: f.write(x + '\n'))

# %%
###########################################################################################
###################################### DECODER ############################################
###########################################################################################

train_generator = F.get_generator_SBND(
    const, G, A, batch_size, EbN0dB_train, type_output, bit_gen, il=INTERLEAVER)

# %% LET THERE BE TRAINING

# Define Callbacks
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='sign_error_logits', mode='min', patience=patience, verbose=0)
TimeStop = F.TimeOut(t0=time.time(), timeout=time_lim_hours)

# training
init_train = time.time()
print('\n\n##############################################################')  
print('                      Start of training')
print('##############################################################\n')
history_decod = decod_model.fit(train_generator, steps_per_epoch=100, epochs=epochs,
                                callbacks=[EarlyStop, TimeStop], verbose=VERBOSE)
print('\nTraining time = {:.3f} hours \n'.format((time.time()-init_train)/3600))

loss, metric, fer_metric = [], [], []
loss.extend(history_decod.history['loss'][:])
metric.extend(history_decod.history['sign_error_logits'][:])
fer_metric.extend(history_decod.history['fer_block_logits'][:])

# save model
decod_model.save(save_path+'decod_model.keras')
decod_model.save(save_path+'decod_model.h5')

# Print error rates into txt file
with open(save_path+'training_metrics.txt', 'w') as f:
    f.write('loss\n')
    for i in range(len(loss)):
        f.write('    {}    {}\\\\ \n'.format(i, loss[i]))
    f.write('\nsign_error_logits\n')
    for i in range(len(loss)):
        f.write('    {}    {}\\\\ \n'.format(i, loss[i]))
    f.write('\nfer_metric_logits\n')
    for i in range(len(loss)):
        f.write('    {}    {}\\\\ \n'.format(i, loss[i]))

# %% Plot training curves

epochs_vec = np.arange(len(loss))
plt.figure()
plt.semilogy(epochs_vec, fer_metric, '-k', label='FER')
plt.semilogy(epochs_vec, loss, '-b', label='BCE')
plt.semilogy(epochs_vec, metric, '-r', label='sign error rate')
plt.title('Training loss and metrics')
plt.grid('on', which='both', ls='--')
plt.xlabel('epoch')
plt.legend()
plt.savefig(save_path+'training_metrics.eps', format='eps', dpi=600)

# %% TEST DECODER

print('\nComputing BER for uncoded')
ber_uncoded, fer_uncoded = [], []

for i_ebn0 in range(len(EbN0dB_test)):
    print('Eb/N0 = {}dB'.format(EbN0dB_test[i_ebn0]))
    sigma2 = 1/(10**(EbN0dB_test[i_ebn0]/10)*const.num_bits_symbol)
    nb_bin_errors, nb_block_errors = 0, 0

    i_sim = 0
    while (nb_block_errors < min_block_errors*10) & (i_sim < nb_sim_max):
        i_sim += 1

        # message generation
        messages = np.random.randint(0, 2, size=(test_batch_size, n))

        # modulation
        mod_messages = F.modulate(const, messages)

        # channel
        noise = np.sqrt(sigma2/2) * (np.random.randn(test_batch_size,nt) + 
                                     1j*np.random.randn(test_batch_size, nt))
        x_noise = mod_messages + noise

        # demodulation
        messages_rec = (F.demodulate_soft(const, x_noise, sigma2) > 0)*1

        # adapt transmitted and received messages to have the correct FER
        messages_rec = messages_rec[:, :k]
        messages = messages[:, :k]

        nb_bin_errors += np.sum((messages != messages_rec)*1)
        nb_block_errors += np.sum(np.sum((messages != messages_rec)*1, axis=1) != 0)

    ber_uncoded.append(nb_bin_errors/(i_sim*k*test_batch_size))
    fer_uncoded.append(nb_block_errors/(i_sim*test_batch_size))

# Compute ber and fer for DNN/RNN decoding

print('\nComputing BER for my decoding')
ber, ber_cw, fer, fer_cw = [], [], [], []

for i_ebn0 in range(len(EbN0dB_test)):
    sigma2 = 1/(10**(EbN0dB_test[i_ebn0]/10)*(k/n)*const.num_bits_symbol)
    nb_bin_errors, nb_bin_errors_cw, nb_block_errors, nb_block_errors_cw = 0, 0, 0, 0

    i_sim = 0
    while (nb_block_errors < min_block_errors) & (i_sim < nb_sim_max) & (time.time()-time_init < max_time*3600):
        i_sim += 1

        # message generation and coding
        messages = np.random.randint(0, 2, size=(test_batch_size, k))
        codewords = (messages@G) % 2

        # interleaver
        if not INTERLEAVER:
            c_inter = codewords
        else:
            idx_inter, idx_deinter = F.interleaver(codewords.shape)
            c_inter = F.reorder(codewords, idx_inter)

        # modulation
        mod_codewords = F.modulate(const, c_inter)

        # channel
        y = mod_codewords+np.sqrt(sigma2/2)*(np.random.randn(test_batch_size, nt)+1j*np.random.randn(test_batch_size, nt))

        # demodulator
        if const.num_bits_symbol == 1:
            llr = -np.real(y)
        else:
            llr = F.demodulate_soft(const, y, sigma2)*sigma2

        if INTERLEAVER:
            llr = F.reorder(llr, idx_deinter)

        # decoder
        # noise_est = F.predict(decod_model,llr).numpy()
        noise_est = decod_model(llr).numpy()

        if type_output == 'denoiser':
            codewords_rec = (np.sign(llr)*noise_est < 0)*1
            messages_rec = codewords_rec@A % 2
        else:
            message_noisy = ((llr > 0)@A % 2)*2-1
            messages_rec = ((message_noisy*noise_est) < 0)*1
            codewords_rec = (messages_rec@G) % 2

        # Add up the errors for each case
        nb_bin_errors += np.sum((messages_rec != messages)*1)
        nb_block_errors += np.sum(np.sum((messages_rec != messages)*1, axis=1) != 0)
        nb_bin_errors_cw += np.sum((codewords_rec != codewords)*1)
        nb_block_errors_cw += np.sum(np.sum((codewords_rec != codewords)*1, axis=1) != 0)

        # Show progress bar in console
        if PROGRESS:
            F.progress(EbN0dB_test[i_ebn0],nb_block_errors, min_block_errors, i_sim)

    # save number of errors found
    with open(save_path+'nb_of_errors.txt', 'a') as file:
        file.write('{}dB -> {} frame errors found in {} simulations of {} frames\n'.format(
            EbN0dB_test[i_ebn0], nb_block_errors, i_sim, test_batch_size))

    if nb_block_errors > 0:
        ber.append(nb_bin_errors/(i_sim*k*test_batch_size))
        fer.append(nb_block_errors/(i_sim*test_batch_size))
        ber_cw.append(nb_bin_errors_cw/(i_sim*n*test_batch_size))
        fer_cw.append(nb_block_errors_cw/(i_sim*test_batch_size))

        print('Eb/N0 = {}dB | #Sims = {} | BER = {:.2e} | FER = {:.2e}'.format(
            EbN0dB_test[i_ebn0], i_sim, ber[-1], fer[-1]))

        # Print error rates into txt file
        with open(save_path+'error_rates.txt', 'w') as f:
            f.write('BER\n')
            for i in range(len(ber)):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_test[i], ber[i]))
            f.write('\nFER\n')
            for i in range(len(fer)):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_test[i], fer[i]))
            f.write('\nBER_cw\n')
            for i in range(len(ber_cw)):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_test[i], ber_cw[i]))
            f.write('\nFER_cw\n')
            for i in range(len(fer_cw)):
                f.write('    {}    {}\\\\ \n'.format(EbN0dB_test[i], fer_cw[i]))

# %% Plot figures

plt.figure()
plt.semilogy(EbN0dB_test, ber_uncoded[0:len(ber)], '--k', label='uncoded')
plt.semilogy(EbN0dB_test, ber, '-ob', label='message BER')
plt.semilogy(EbN0dB_test, ber_cw, '--b', label='codeword BER')
plt.title('BER {} ({},{}) - {}{} - {}'.format(TYPE, n, k, 2**MOD_ORDER, MOD_TYPE, type_output))
plt.xlabel('Eb/N0 (dB)')
plt.grid("on", which='both', ls='--')
plt.savefig(save_path+'BER.eps', format='eps', dpi=600)

plt.figure()
plt.semilogy(EbN0dB_test, fer_uncoded[0:len(fer)], '--k', label='uncoded')
plt.semilogy(EbN0dB_test, fer, '-ob', label='message BER')
plt.semilogy(EbN0dB_test, fer_cw, '--b', label='codeword BER')
plt.title('FER {} ({},{}) - {}{} - {}'.format(TYPE, n, k, 2**MOD_ORDER, MOD_TYPE, type_output))
plt.xlabel('Eb/N0 (dB)')
plt.grid("on", which='both', ls='--')
plt.savefig(save_path+'FER.eps', format='eps', dpi=600)
# plt.show()
