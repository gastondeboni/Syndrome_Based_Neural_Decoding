#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
author: Gastón De Boni Rovella
contact: gdebonirovella@gmail.com (English, Spanish or French)

This file contains some functions necessary to reproduce the results
from the article "Scalable Syndrome-based Neural Decoders for Bit-Interleaved
Coded Modulations", IEEE ICMLCN 2024.
'''

# Imports
import numpy as np
import tensorflow as tf
from commpy import PSKModem, QAMModem
from time import time
from itertools import combinations

#%%############################################################################
######################### NEURAL ARCHITECTURES ################################
###############################################################################

################################### MLP #######################################

# this function converts a vector of LLRs to a concatenation of its absolute 
# value and its syndrome.
class llr_to_abs_syn(tf.keras.layers.Layer):

    def __init__(self, H, f_abs=1, f_syn=1, name='Preprocessing', **kwargs):
        self.H = tf.cast(H,tf.float32)
        self.f_abs = f_abs
        self.f_syn = f_syn
        super().__init__(name=name, **kwargs)
  
    def build(self, input_shape):    
        super().build(input_shape)
  
    def call(self, llr):
        syndrome =  tf.matmul(tf.cast((llr>0),tf.float32), self.H, transpose_b=True) % 2
        llr_s = tf.concat([tf.math.abs(llr)*self.f_abs, (syndrome*-2.0+1)*self.f_syn], axis=1)
        return llr_s

def get_MLP_decoder(mode, H, size_factor, activation='linear', f_abs=1, f_syn=1, depth=10):
    n = H.shape[1]
    k = H.shape[1] - H.shape[0]
    
    if mode == 'denoiser':
        output_size = n
    elif mode == 'decoder':
        output_size = k

    # input to the network
    inputs = tf.keras.Input(shape=(n), name='Input')
    
    # preprocessing
    in_layer = llr_to_abs_syn(H, f_abs, f_syn)(inputs)
    
    # MLP
    h = tf.keras.layers.Dense(units=size_factor*(2*n-k), activation='relu', name='dense0')(in_layer)
    for i in range(1,depth+1):
        # aux = tf.keras.layers.Concatenate()([h, in_layer])
        h = tf.keras.layers.Dense(units=size_factor*(2*n-k), activation='relu', name=f'dense{i}')(tf.concat([in_layer,h],axis=-1))
        
    outs = tf.keras.layers.Dense(output_size, activation=activation, name='outs')(tf.concat([in_layer,h],axis=-1))
    return tf.keras.Model(inputs=inputs, outputs=outs, name='decoder')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class llr_to_RNN(tf.keras.layers.Layer):

    def __init__(self, H, RNN_steps, f_abs=1, f_syn=1, name='rnnPreprocessing', **kwargs):
        self.RNN_steps = RNN_steps
        self.H = tf.cast(H,tf.float32)
        self.f_abs = f_abs
        self.f_syn = f_syn
        super().__init__(name=name, **kwargs)
  
    def build(self, input_shape):
        super().build(input_shape)
  
    def call(self, llr):
        syndrome =  tf.matmul(tf.cast((llr>0),tf.float32), self.H, transpose_b=True) % 2
        llr_s = tf.concat([tf.math.abs(llr)*self.f_abs, (syndrome*-2.0+1)*self.f_syn], axis=1)
        return tf.repeat(llr_s[:,None,:], repeats=self.RNN_steps, axis=1)

def get_RNN_decoder(mode, H, RNN_steps, RNN_size_factor, activation='linear', f_abs=1, f_syn=1, depth=5):
    n = H.shape[1]
    k = H.shape[1] - H.shape[0]
    
    if mode == 'denoiser':
        output_size = n
    elif mode == 'decoder':
        output_size = k

    # input to the network
    input_rnn = tf.keras.Input(shape=(n), name='Input')
    
    # preprocessing in module and syndrome
    h = llr_to_RNN(H, RNN_steps, f_abs, f_syn)(input_rnn)
    
    # RNN 
    #h = tf.keras.layers.LayerNormalization()(h)
    for i in range(depth-1):
        h = tf.keras.layers.GRU(units=RNN_size_factor*(2*n-k), return_sequences=True, name=f'gru{i}')(h)
    h = tf.keras.layers.GRU(units=RNN_size_factor*(2*n-k), name='gru_final')(h)
    outs = tf.keras.layers.Dense(output_size, activation=activation, name='outs')(h)
    return tf.keras.Model(inputs=input_rnn, outputs=outs, name='decoder')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ECCT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Error Correction Code Transformer
def get_ECCT(mode, d_model, nb_encoder, nb_head, H, mask=True):
    
    n, k = H.shape[1], H.shape[1]-H.shape[0]
    
    if mode == 'denoiser':
        output_size = n
    elif mode == 'decoder':
        output_size = k
    
    if mask:
        mask = tf.cast(create_mask(H), tf.float32)
    else:
        mask = None
    
    layer_input = tf.keras.Input(shape=(n), name='Input')
    layer_preprocessing = llr_to_abs_syn(H)(layer_input)
    embedding = EmbeddingLayer(d_model)(layer_preprocessing)
    enc_output = Encoder(nb_encoder=nb_encoder, d_model=d_model, nb_head=nb_head)(embedding,mask)
    final_output = FinalLayer(output_size)(enc_output)
    model = tf.keras.Model(layer_input, final_output, name='ECCT')
    
    return model

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% r-ECCT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_recurrent_ECCT(mode, d_model, steps, nb_head, H, nb_dense_layers=2, size_factor_dense=4, mask=True):
    
    n, k = H.shape[1], H.shape[1]-H.shape[0]
    
    if mode == 'denoiser':
        output_size = n
    elif mode == 'decoder':
        output_size = k
        
    if mask:
        mask = tf.cast(create_mask(H), tf.float32)
    else:
        mask = None
    
    layer_input = tf.keras.Input(shape=(n), name='Input')
    layer_preprocessing = llr_to_abs_syn(H)(layer_input)
    embedding = EmbeddingLayer(d_model)(layer_preprocessing)
    enc_output = RecurrentEncoder(steps, d_model, nb_head,nb_dense_layers,size_factor_dense)(embedding,mask)
    final_output = FinalLayer(output_size)(enc_output)
    
    return tf.keras.Model(layer_input, final_output, name='r-ECCT')

#%%%%%%%%%%%%%%%%%%%%%%%%%%% Double-masking ECCT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_double_ECCT(mode, d_model, nb_encoder, nb_head, H1, H2):
    
    n, k = H1.shape[1], H1.shape[1]-H1.shape[0]
    
    if mode == 'denoiser':
        output_size = n
    elif mode == 'decoder':
        output_size = k
    
    mask1 = tf.cast(create_mask(H1), tf.float32)
    mask2 = tf.cast(create_mask(H2), tf.float32)
    
    # input
    layer_input = tf.keras.Input(shape=(n), name='Input')
    
    # Path 1
    layer_preprocessing_1 = llr_to_abs_syn(H1, name='Preprocessing1')(layer_input)
    embedding_1 = EmbeddingLayer(d_model, name='Embedding1')(layer_preprocessing_1)
    enc_output_1 = Encoder(nb_encoder=nb_encoder, d_model=d_model, nb_head=nb_head, name='Encoder1')(embedding_1,mask1)
    
    # Path 2
    layer_preprocessing_2 = llr_to_abs_syn(H2, name='Preprocessing2')(layer_input)
    embedding_2 = EmbeddingLayer(d_model, name='Embedding2')(layer_preprocessing_2)
    enc_output_2 = Encoder(nb_encoder=nb_encoder, d_model=d_model, nb_head=nb_head, name='Encoder2')(embedding_2,mask2)
    
    # concatenation and output
    layer_concat = tf.keras.layers.Concatenate(name='Concatenate')([enc_output_1, enc_output_2])
    final_output = FinalLayer(output_size)(layer_concat)
    model = tf.keras.Model(layer_input, final_output, name='Double-masking-ECCT')
    
    return model

#%%############################################################################
####################### GENERATOR FOR TRAINING ################################
###############################################################################
        
def get_generator_SBND(const, G, A, batch_size, ebn0_lim, mode, bit_gen, clip_llr=None, il=False):
    k,n = G.shape
    
    u = np.zeros(shape=(batch_size,k), dtype='int')
    if bit_gen == '01':
        u = find_message_01(G)
        u = np.repeat(u[None,:], batch_size, axis=0)
    elif bit_gen == 'zeros':
        u = np.zeros(shape=(batch_size,k), dtype='int')
    elif bit_gen == 'ones':
        u = np.ones((batch_size,k), dtype='int')
        
    c = (u@G) % 2
    
    while True:
        EbN0 = np.random.uniform(ebn0_lim[0], ebn0_lim[1])
        sigma2 = 1/((k/n)*10**(EbN0/10)*const.num_bits_symbol)
        
        if bit_gen == 'random':
            u = np.random.randint(0,2, size=(batch_size,k), dtype=int)
            c = (u@G) % 2
        
        # interleaver
        if not il:
            c_inter = c[:]
        else:
            idx_inter, idx_deinter = interleaver(c.shape)
            c_inter = np.take_along_axis(c, idx_inter, axis=1)
        
        # modulation
        x = modulate(const,c_inter)

        # channel
        noise =  np.sqrt(sigma2/2)* (np.random.randn(batch_size,int(n/const.num_bits_symbol)) +1j*np.random.randn(batch_size,int(n/const.num_bits_symbol)))
        y = x + noise
        
        # demodulation            
        if const.num_bits_symbol == 1:
            llr = -np.real(y)
        else:
            llr = demodulate_soft(const, y, sigma2, clip_llr)*sigma2
            
        # deinterleaver
        if il:
            llr = np.take_along_axis(llr, idx_deinter, axis=1)
        
        # decoding        
        if mode == 'decoder':
            ut = (llr>0)@A %2
            z = (ut != u)*1.0
        elif mode == 'denoiser':
            z = ((llr>0)*1 != c)*1.0

        yield (llr, z)

#%%############################################################################
####################### MODULATION AND DEMODULATION ###########################
###############################################################################

def get_modulator(MOD_TYPE, MOD_ORDER):
    if MOD_TYPE == 'PSK':
        const = PSKModem(2**MOD_ORDER)
    elif MOD_TYPE == 'QAM':
        const = QAMModem(2**MOD_ORDER)
        
    power_mean = np.mean(const.constellation.real**2 + const.constellation.imag**2)
    if  abs(power_mean-1) > 0.00001:    
        const._constellation = const._constellation/np.sqrt(power_mean)
    return const

def modulate(const, input_bits):
    
    # for speed, if BPSK return this directly
    if const.num_bits_symbol == 1:
        return 1.-2.*input_bits
    
    # Number of codewords to modulate
    batch_size = input_bits.shape[0]
    
    # reshape and convert to integers
    input_reshape = np.reshape(input_bits, (batch_size* input_bits.shape[1]//const.num_bits_symbol, const.num_bits_symbol))
    input_dec = bit2dec_array(input_reshape).astype(int)
    
    #modulate
    symbols = const._constellation[input_dec]
    
    return np.reshape(symbols, (batch_size, input_bits.shape[1]//const.num_bits_symbol))
   
def demodulate_soft(const, input_symbols, noise_var=0, clip=None):

    batch_size = input_symbols.shape[0]
    demod_bits = np.zeros((batch_size, input_symbols.shape[1] * const.num_bits_symbol))
    
    for i in np.arange(input_symbols.shape[1]):
        current_symbol = input_symbols[:,i]
        for bit_index in np.arange(const.num_bits_symbol):
            llr_num = np.zeros(batch_size)
            llr_den = np.zeros(batch_size)
            for bit_value, symbol in enumerate(const._constellation):
                if (bit_value >> bit_index) & 1:
                    llr_num += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
                else:
                    llr_den += np.exp((-abs(current_symbol - symbol) ** 2) / noise_var)
            demod_bits[:,i * const.num_bits_symbol + const.num_bits_symbol - 1 - bit_index] = np.log(llr_num / llr_den)

    if clip is not None:
        demod_bits = np.clip(demod_bits, -clip, clip)
    return demod_bits

def bit2dec_array(in_bitarray):
    
    numbers = np.zeros(in_bitarray.shape[0])

    for i in range(in_bitarray.shape[1]):
        numbers = numbers + in_bitarray[:,i] * pow(2, in_bitarray.shape[1] - 1 - i)

    return numbers
   
#%%############################################################################ 
############## FUNCTIONS FOR CONSTRUCTION OF POLAR CODES ######################
###############################################################################

def KernelPower(n,T):
    pow_matrix = T
    for i in range(1,n):
        pow_matrix = np.kron(pow_matrix, T)
    return pow_matrix 

def Construct_PC(myPC):
    print('\nConstructing polar code with dimensions ({},{})'.format(myPC.N,myPC.K))
    
    lookup = myPC.frozen_lookup
    P = KernelPower(int(np.log2(myPC.N)), np.array([[1,0],[1,1]]))
    
    #P[[0,10],:] = P[[10,0],:]
    
    G = P[lookup==1,:]
    H = P[:,lookup==0].T
    V = np.eye(myPC.N)[lookup==1,:]
    
    if np.sum(np.mod(G@H.T,2)) != 0:
        raise ValueError("Parity-check matrix condition not satisfied")
        
    print('---> PC matrix condition verified\n')
    return G,H,P,V

def H_to_G_A(H):
        
    n = H.shape[1] 
    P = KernelPower(int(np.log2(n)), np.array([[1,0],[1,1]]))
    
    lookup = np.sum(P@H.T % 2, axis=1) == 0
    
    # Compute matrices
    G = P[lookup,:]
    V = np.eye(n)[lookup,:]
    
    # inverse matrix to go from codeword to message
    A = P @ V.T %2
    
    return G, A

def H_2_frozenlookup(H):
    if H.shape[0] > H.shape[1]:
        H = H.T
        
    n = H.shape[1] 
    P = KernelPower(int(np.log2(n)), np.array([[1,0],[1,1]]))
    
    frozen_lookup = np.sum(P@H.T % 2, axis=1)%2 == 1
    
    return frozen_lookup

#%%############################################################################ 
############### FUNCTIONS FOR CONSTRUCTION OF BCH CODES #######################
###############################################################################

def octal_to_bin(octal):
    octal_str = str(octal)
    binary = ''
    
    for c in octal_str:
        if c == '0':
            binary += '000'
        elif c == '1':
            binary += '001'
        elif c == '2':
            binary += '010'
        elif c == '3':
            binary += '011'
        elif c == '4':
            binary += '100'
        elif c == '5':
            binary += '101'
        elif c == '6':
            binary += '110'
        elif c == '7':
            binary += '111'
            
    return str(int(binary))
            
def BCH_gen_matrix(octal,n,k):
    
    binary = octal_to_bin(octal)
    l = len(binary)
    vector = np.zeros(shape=l)
    for i in range(l):
        vector[i] = int(binary[i])
        
    G = np.zeros(shape=(k,n))
    for i in range(G.shape[0]):
        G[i,i:i+l] = vector
        
    return G

            
def BCH_systematic(octal,n,k):
    
    G = to_standard(BCH_gen_matrix(octal, n, k), side='left')
    C = G[:,k:]
    H = np.concatenate((C.T,np.eye(n-k)), axis=1)
    
    return G, H

#%%############################################################################ 
################## PARITY-CHECK MATRIX AND LINEAR CODES #######################
###############################################################################

def get_systematic_code(H2, PCM_FORM):
    #standardize matrix
    H = np.matrix.copy(H2)
    Hs = to_standard(H, side='left')
    
    idx_sys = np.zeros(H.shape[1]) > 1
    i = 0
    col_list = []
    while np.sum(idx_sys) < H.shape[0]:
        if np.sum(Hs[:,i]) == 1 and not col_list:
            idx_sys[i] = True
            col_list.append(Hs[:,i])
        elif np.sum(Hs[:,i]) == 1 and not np.any(np.all(Hs[:,i] == col_list, axis=1)):
            idx_sys[i] = True
            col_list.append(Hs[:,i])
        i += 1

    # Construct generator matrix accordingly
    Gd = np.concatenate((np.eye(H.shape[1]-H.shape[0]), Hs[:,~idx_sys].T), axis=1).astype(int)
    
    # Adapt original matrix
    H = np.concatenate((H[:,~idx_sys],H[:,idx_sys]), axis=1)
    
    # choose form of PC matrix
    H = change_H(H, PCM_FORM)
        
    # get systematic indices from the generator matrix
    sys_idx = np.arange(Gd.shape[1])<Gd.shape[0]
    
    if np.sum(Gd@H.T %2) != 0:
        raise ValueError('PC matrix criterion not satisfied -> Invalid (G,H) pair!')
    if np.sum(sys_idx) != Gd.shape[0]:
        raise ValueError('There are NOT k systematic bits')
        
    return Gd, H, sys_idx

def change_H(A, matrix_type):
    
    if matrix_type == 'normal':
        return A
    if matrix_type == 'standright':
        return to_standard(A, side='right')
    if matrix_type == 'standleft':
        return to_standard(A, side='left')
    if matrix_type == 'random':
        return randomize_H(A)
    if 'sparsify' in matrix_type:
        As = to_standard(A, side='right')
        return reduce_weight(As, order=int(matrix_type[-1]))
    if matrix_type == 'sparsify_mask':
        A2 = reduce_weight_connectivity(to_standard(A, side='right'))
        A3 = reduce_weight_connectivity(A)
        if np.sum(create_connectivity_matrix(A2)) < np.sum(create_connectivity_matrix(A3)):
            return A2
        else:
            return A3

def to_standard(A2, side):
    A = np.matrix.copy(A2)
    
    if side == 'left':
        i, i1 = 0, 0

        while (i1 < A.shape[0]):
            
            swap_ind = i1+1
            while (A[i1,i] == 0) and (swap_ind < A.shape[0]):
                
                if A[swap_ind,i] == 1:
                    A[[i1,swap_ind],:] = A[[swap_ind,i1],:]
                swap_ind += 1
                
            if A[i1,i] == 1:
                for j in range(A.shape[0]):
                    if j != i1:
                        A[j,:] = (A[j,:]+A[i1,:]*A[j,i])%2
                i1 += 1
            i += 1
            
    elif side == 'right':
        i, i1 = A.shape[1]-1, A.shape[0]-1
        
        while (i1 >= 0):
            
            swap_ind = i1-1
            while (A[i1,i] == 0) and (swap_ind >= 0):
                
                if A[swap_ind,i] == 1:
                    A[[i1,swap_ind],:] = A[[swap_ind,i1],:]
                swap_ind -= 1
                
            if A[i1,i] == 1:
                for j in range(A.shape[0]):
                    if j != i1:
                        A[j,:] = (A[j,:]+A[i1,:]*A[j,i])%2
                i1 -= 1
            i -= 1
        
    return A

def randomize_H(H2):
    H = np.matrix.copy(H2)
    if H.shape[0] > H.shape[1]:
        H = H.T
    
    for i in range(5000):
        
        idx = np.random.randint(0,H.shape[0], size=2)
        while idx[0] == idx[1]:
            idx = np.random.randint(0,H.shape[0], size=2)

        H_aux = np.matrix.copy(H)
        H[idx[0],:] = np.mod(H_aux[idx[0],:] + H_aux[idx[1],:],2)
        
    return H

def reduce_weight(H2, order):
    # initialization
    H = np.matrix.copy(H2)
    
    positions = np.arange(H.shape[0])
    pos_list = list()
    for i in range(1,order+1):
        pos_list.extend( (np.array(list(combinations(positions,i)))) )
    
    # algorithm
    FoundBetter = True
    while FoundBetter:
        FoundBetter = False
        w_min = H.shape[1]
        for i in range(H.shape[0]):
            w_i = np.sum(H[i])
            pos_list_filter = [x for x in pos_list if i not in x]
            for pos in pos_list_filter:
                w_new = np.sum( (H[i] + np.sum(H[pos], axis=0)) %2 )   
                if w_new<min(w_min, w_i):
                    w_min = w_new
                    i_best, pos_best = i, pos
                    FoundBetter = True
        if FoundBetter:
            H[i_best] = (H[i_best] + np.sum(H[pos_best], axis=0)) %2
            # print(np.sum(H))
            print(f'\rweight = {np.sum(H)}', end='\r')
    
    return H

# def reduce_weight(H2):
#     # initialization
#     H = np.matrix.copy(H2)
#     FoundBetter = True
    
#     # algorithm
#     while FoundBetter:
#         FoundBetter = False
#         w_min = H.shape[1]
#         for i in range(H.shape[0]):
#             w_i = np.sum(H[i])
#             for j in range(H.shape[0]):
#                 w_new = np.sum((H[j]+H[i]) % 2)
#                 if w_new<min(w_min, w_i) and (i != j):
#                     w_min = w_new
#                     i_best, j_best = i, j
#                     FoundBetter = True
#         if FoundBetter:
#             H[i_best] = (H[i_best]+H[j_best]) %2
    
#     return H

def create_connectivity_matrix(H):
    l, n = H.shape
    k = n-l
    mask = np.eye(n)
    for i in range(n-k):
        idx = np.argwhere(H[i,:] == 1).T[0]
        for k in idx:
            mask[k,idx] = 1
            mask[idx,k] = 1
    return mask.astype(int)

def reduce_weight_connectivity(H2):
    H = np.matrix.copy(H2)

    while 1:
        
        lowest_weight = np.sum(create_connectivity_matrix(H))
        idx = list(range(H.shape[0]))
        np.random.shuffle(idx)
        
        for i in idx:
            
            idx2 = list(range(H.shape[0]))
            np.random.shuffle(idx2)
            for j in idx2:
                H_new = np.matrix.copy(H)
                H_new[i] = (H[j]+H[i]) % 2
                new_weight = np.sum(create_connectivity_matrix(H_new))
                if (new_weight < lowest_weight) and (i != j):
                    H[i] = (H[i]+H[j]) %2
                    break
            
            if j != idx2[-1]:
                break
        
        if i == idx[-1]:
            return H

#%%############################################################################ 
################ COMPUTING THE MAXIMUM LIKELIHOOD BOUND #######################
###############################################################################

def get_dist_llr(llr, codewords):
    llr_sign = llr * (-2.*codewords + 1.) # apply BPSK mapping
    return np.mean( np.log(1. + np.exp(llr_sign)) , axis=1)

# ML bound based on LLR
def ML_bound_decoder_llr(codewords, x_noise_llr, codewords_rec):

    # indices where there was decoding failure
    idx_dec_errors = np.sum((codewords_rec!=codewords)*1, axis=1) != 0
    
    # distances d(x_noise, MyDec) and d(x_noise, PerfectDec)
    dist_mydec2rec = get_dist_llr(x_noise_llr[idx_dec_errors], codewords_rec[idx_dec_errors])
    dist_correct2rec = get_dist_llr(x_noise_llr[idx_dec_errors], codewords[idx_dec_errors])
    
    # idx where the SCL received codeword is MORE probable than perfect decoding
    # we assume that in those ids, ML would have gotten it right
    idx_mydec_wins = np.argwhere(idx_dec_errors).T[0][dist_correct2rec > dist_mydec2rec]
    
    # and finally choose the ML-bound received codewords 
    codewords_rec_MLB = np.matrix.copy(codewords)
    codewords_rec_MLB[idx_mydec_wins] = codewords_rec[idx_mydec_wins]
    
    return codewords_rec_MLB

# ML bound considering modulations (much slower)
def ML_bound_decoder_symbol(const, codewords, x_noise, codewords_rec):

    # indices where there was decoding failure
    idx_dec_errors = np.sum((codewords_rec!=codewords)*1, axis=1) != 0
    
    # distances d(x_noise, MyDec) and d(x_noise, PerfectDec)
    dist_mydec2rec = np.sum(np.abs(modulate(const,codewords_rec[idx_dec_errors]) - x_noise[idx_dec_errors])**2, axis=1)
    dist_correct2rec = np.sum(np.abs(modulate(const,codewords[idx_dec_errors]) - x_noise[idx_dec_errors])**2, axis=1)
    
    # idx where the codeword_rec is MORE probable than perfect decoding
    # we assume that in those ids, ML would have gotten it right
    idx_mydec_wins = np.argwhere(idx_dec_errors).T[0][dist_correct2rec > dist_mydec2rec]
    
    # and finally choose the ML-bound received codewords 
    codewords_rec_MLB = np.matrix.copy(codewords)
    codewords_rec_MLB[idx_mydec_wins] = codewords_rec[idx_mydec_wins]
    
    return codewords_rec_MLB
   

#%%############################################################################
######################## OTHER AUXILIARY FUNCTIONS ############################
###############################################################################

# Two functions that serve as metrics during training
def sign_error_logits(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.abs(y_true*2.0-1 - tf.math.sign(y_pred))/2.0)

def fer_block_logits(y_true, y_pred):
    return tf.math.reduce_mean(tf.cast(tf.math.reduce_sum( tf.math.abs(y_true*2.0-1 - tf.math.sign(y_pred)), axis=1 ) != 0, dtype=tf.float32) )

# for bit interleaving
def interleaver(shape):
    idx = np.repeat(np.arange(shape[1])[None,:], shape[0], axis=0)
    for i in range(shape[0]):
        np.random.shuffle(idx[i])
    return idx, idx.argsort()

# to measure progress during Bit Error Rate testing
def progress(EbN0dB, current, total, sim, min_sim=1, bar_length=30):
    fraction = min(current / total,1)
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current >= total and sim >= min_sim else '\r'
    print(f'\rEb/N0 = {EbN0dB}dB: [{arrow}{padding}] {current}/{total} BEs found | {sim} sims', end=ending)
    
# inference function with tensorflow decorator for speed
@tf.function
def predict(model, inputs):
    return model(inputs)

# time out callback for training
class TimeOut(tf.keras.callbacks.Callback):
    def __init__(self, t0, timeout):
        super().__init__()
        self.t0 = t0
        self.timeout = timeout  # time in hours

    def on_train_batch_end(self, batch, logs=None):
        if time() - self.t0 > self.timeout * 3600:
            print('\n\n##############################################################\n')
            print(f"\nReached {(time() - self.t0) / 3600:.3f} hours of training, stopping\n\n")
            self.model.stop_training = True

# this function, given a generator matrix G, finds a message that its
# respective codeword has approximately the same number of zeros and ones
def find_message_01(G):
    k,n = G.shape
    
    u = np.zeros(shape=k, dtype=int)
    cw_weight = 0
    for i in range(k):
        u[i] = (u[i] + 1) % 2
        c = u@G % 2
        new_weight = np.sum(c)
        
        if abs(new_weight-n/2) < 1:
            return u
        elif abs(new_weight-n/2) <= abs(cw_weight-n/2):
            continue
        else:
            u[i] = (u[i] + 1) % 2
    
    return u

#%%############################################################################
####################### TRANSFORMER-SPECIFIC FUNCTIONS ########################
###############################################################################  

## Some auxiliary functions

def create_mask(H):
    l, n = H.shape
    k = n-l
    mask = np.eye(2*n-k)
    for i in range(n-k):
        idx = np.argwhere(H[i,:] == 1).T[0]
        mask[n+i,idx] = 1
        mask[idx,n+i] = 1
        for k in idx:
            mask[k,idx] = 1
            mask[idx,k] = 1
    return mask.astype(int)

# should be the same as create_mask
# def build_mask(H):
#     l, n = H.shape
#     k = n-l
#     mask = np.eye(2*n-k)
#     for i in range(n-k):
#         idx = np.argwhere(H[i,:] == 1).T[0]
#         for j in idx:
#             for k in idx:
#                 mask[j, k] = 1
#                 mask[k, j] = 1
#                 mask[n + i, j] = 1
#                 mask[j, n + i] = 1
#     return mask
        
#%%############################################################################
############## LAYERS TO BUILD TRANSFORMER-BASED ARCHITECTURES ################
###############################################################################   

# Embedding layer
class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, name='EmbeddingLayer', **kwargs):
        self.d_model = d_model
        super().__init__(name=name, **kwargs) 
  
    def build(self, input_shape):
        #self.embedding = self.add_weight("embedding",shape=[int(input_shape[-1]),self.d_model])
        w_init = tf.random_normal_initializer()
        self.embedding = tf.Variable(name='embedding', initial_value=w_init(shape=(input_shape[-1], self.d_model), dtype=tf.float32),trainable=True)
        super().build(input_shape)
  
    def call(self, x):
        embed = tf.expand_dims(self.embedding,axis=0) * tf.expand_dims(x,axis=-1)
        return embed


# Test for Embedding Layer
'''
n, k, d_model = 8, 4, 16
layer_input = tf.keras.Input(shape=(2*n-k))
embedding = EmbeddingLayer(d_model)(layer_input)
model = tf.keras.Model(layer_input, embedding)
model.summary()

a = tf.Variable(tf.zeros(2*n-k))
a = a[2].assign(1)
print(a.numpy())
print(a.shape)
print(model.layers[0](a).shape)
print(model.layers[1](model.layers[0](a)).shape)
'''

class myMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, nb_head, **kwargs):
        assert d_model % nb_head == 0
        self.dim = d_model
        self.head_dim = d_model // nb_head
        self.nb_head = nb_head
        super(**kwargs).__init__()
    
    def build(self, input_shape):
        self.query_layer = tf.keras.layers.Dense(self.dim)
        self.value_layer = tf.keras.layers.Dense(self.dim)
        self.key_layer = tf.keras.layers.Dense(self.dim)
        self.softmax = tf.keras.layers.Softmax()
        self.out_proj = tf.keras.layers.Dense(self.dim)
        super().build(input_shape)

    # def mask_softmax(self, x, mask):
    #     x_expe = tf.math.exp(x)
    #     x_expe_masked = x_expe * tf.cast(mask, 'float32')
    #     x_expe_sum = tf.reduce_sum(x_expe_masked, axis=-1)    
    #     x_expe_sum = tf.expand_dims(x_expe_sum, axis=-1)
    #     softmax = x_expe_masked / x_expe_sum
    #     return softmax


    def call(self, x, mask):

        in_query, in_key, in_value = x

        Q = self.query_layer(in_query)
        K = self.key_layer(in_key)
        V = self.value_layer(in_value)

        batch_size = tf.shape(Q)[0]
        Q_seq_len = tf.shape(Q)[1]
        K_seq_len = tf.shape(K)[1]
        V_seq_len = tf.shape(V)[1]

        Q = tf.reshape(Q, [batch_size, Q_seq_len, self.nb_head, self.head_dim])
        K = tf.reshape(K, [batch_size, K_seq_len, self.nb_head, self.head_dim])
        V = tf.reshape(V, [batch_size, V_seq_len, self.nb_head, self.head_dim])
        
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])
        
        # Scaled dot product attention
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK / tf.math.sqrt(tf.cast(self.dim, dtype=tf.float32))
        
        if mask is not None:
            softmax_QK = self.softmax(QK, mask==1)
        else:
            softmax_QK = tf.nn.softmax(QK, axis=-1)
        
        attention = tf.matmul(softmax_QK, V)
        attention = tf.transpose(attention, [0, 2, 1, 3])
        
        # Concat
        attention = tf.reshape(attention, [batch_size, Q_seq_len, self.nb_head*self.head_dim])
        out_attention = self.out_proj(attention)
        
        return out_attention

# Test for Multi-Head Self-Attention
'''
n, k, d_model, nb_head = 8, 4, 16, 4
H = np.random.randint(0,2, size=(n-k,n))
mask = tf.cast(create_mask(H), tf.float32)

layer_input = tf.keras.Input(shape=(2*n-k))
embedding = EmbeddingLayer(d_model)(layer_input)
attention = myMultiHeadAttention(d_model,nb_head)((embedding, embedding, embedding), mask=mask)

model = tf.keras.Model(layer_input, attention)
model.summary()

a = tf.Variable(tf.zeros(2*n-k))
a = a[2].assign(1)
print(a.numpy())
print(a.shape)
print(model.layers[0](a).shape)
embed = model.layers[1](model.layers[0](a))
print(embed.shape)
print(model.layers[2]((embed,embed,embed), mask).shape)
'''

# Note: Multi-Head Attention can be implemented either with the built-in function
# or with the custom function MyMultiHeadAttention - they both do exctly the same
# and in the same time.

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, nb_head, nb_dense_layers=2, size_factor_dense=4, **kwargs):
        self.d_model = d_model
        self.nb_head = nb_head
        self.nb_dense_layers = nb_dense_layers
        self.size_factor_dense = size_factor_dense
        super(**kwargs).__init__()
  
    def build(self, input_shape):
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        # self.multi_head_attention = myMultiHeadAttention(self.d_model, self.nb_head)
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=self.nb_head,key_dim=self.d_model//self.nb_head)
        
        # dense layers
        self.dense_layers = []
        for i in range(self.nb_dense_layers-1):
            self.dense_layers.append(tf.keras.layers.Dense(self.d_model*self.size_factor_dense, activation='gelu'))
        self.dense_layers.append(tf.keras.layers.Dense(self.d_model, activation='gelu'))
        super().build(input_shape)
        
    def call(self, x, mask):
        
        # 1st stage
        x_norm = self.norm1(x)
        # attention = self.multi_head_attention((x_norm, x_norm, x_norm), mask)
        attention = self.multi_head_attention(x_norm, x_norm, x_norm, attention_mask=mask)
        out1 = x + attention
        
        # 2nd stage
        h = self.norm2(out1)
        for lay in self.dense_layers:
            h = lay(h)
        
        return h + out1

# Test for Encoder Layer
'''
n, k, d_model, nb_head = 8, 4, 16, 4
H = np.random.randint(0,2, size=(n-k,n))
mask = tf.cast(create_mask(H), tf.float32)

layer_input = tf.keras.Input(shape=(2*n-k))
embedding = EmbeddingLayer(d_model)(layer_input)
enc_output = EncoderLayer(d_model,nb_head)(embedding, mask=mask)

model = tf.keras.Model(layer_input, enc_output)
model.summary()

a = tf.Variable(tf.zeros(2*n-k))
a = a[2].assign(1)
print(a.numpy())
print(a.shape)
print(model.layers[0](a).shape)
print(model.layers[1](model.layers[0](a)).shape)
print(model.layers[2](model.layers[1](model.layers[0](a)), mask).shape)
'''

class Encoder(tf.keras.layers.Layer):

    def __init__(self, nb_encoder, d_model, nb_head, name='EncodingLayer', **kwargs):
        self.nb_encoder = nb_encoder
        self.d_model = d_model
        self.nb_head = nb_head
        super().__init__(name=name, **kwargs)
  
    def build(self, input_shape):
        self.encoder_layers = []
        for nb in range(self.nb_encoder):
            self.encoder_layers.append(EncoderLayer(self.d_model, self.nb_head))
        super().build(input_shape)
            
    def call(self, x, mask):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x

class FinalLayer(tf.keras.layers.Layer):

    def __init__(self, output_size, name='DecodingLayer', **kwargs):
        self.output_size = output_size
        super().__init__(name=name, **kwargs)
  
    def build(self, input_shape):
        self.norm = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(1, activation='linear')
        self.squeeze = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-1))
        self.dense2 = tf.keras.layers.Dense(self.output_size, activation='linear')
        super().build(input_shape)
        
    def call(self, x):
        x_norm = self.norm(x)
        dense_1st = self.dense1(x_norm)
        squeezed = self.squeeze(dense_1st)
        dense_2nd = self.dense2(squeezed)
        return dense_2nd

# Test of the full architecture
'''
n, k, d_model, nb_head, nb_encoder = 8, 4, 16, 4, 5
H = np.random.randint(0,2, size=(n-k,n))
mask = tf.cast(create_mask(H), tf.float32)

layer_input = tf.keras.Input(shape=(2*n-k))
embedding = EmbeddingLayer(d_model)(layer_input)
enc_output = Encoder(nb_encoder,d_model,nb_head)(embedding, mask=mask)
final_out = FinalLayer(output_size=k)(enc_output)

model = tf.keras.Model(layer_input, final_out)
model.summary()

a = tf.Variable(tf.zeros(2*n-k))
a = a[2].assign(1)
print(a.numpy())
print(a.shape)
print(model.layers[0](a).shape)
print(model.layers[1](model.layers[0](a)).shape)
print(model.layers[2](model.layers[1](model.layers[0](a)), mask).shape)
print(model.layers[3](model.layers[2](model.layers[1](model.layers[0](a)), mask)).shape)
'''

class RecurrentEncoder(tf.keras.layers.Layer):

    def __init__(self, steps, d_model, nb_head, nb_dense_layers=2, size_factor_dense=4, name='RecEncoder', **kwargs):
        self.steps = steps
        self.d_model = d_model
        self.nb_head = nb_head
        self.nb_dense_layers = nb_dense_layers
        self.size_factor_dense = size_factor_dense
        super().__init__(name=name,**kwargs)
  
    def build(self, input_shape):
        self.encoder = EncoderLayer(self.d_model,self.nb_head,self.nb_dense_layers,self.size_factor_dense)
        super().build(input_shape)
            
    def call(self, x, mask):
        for i in range(self.steps):
            x = self.encoder(x, mask)
        return x



























    
