
#pragma once

#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>

#include "AESObject.h"

AESObject::AESObject(char* filename) {

    std::ifstream f(filename);
    std::string str {
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>()
    };
	f.close();
	int len = str.length();
	char common_aes_key[len+1];
	memset(common_aes_key, '\0', len+1);
	strcpy(common_aes_key, str.c_str());

    if(!(ctx = EVP_CIPHER_CTX_new())) {
        ERR_print_errors_fp(stderr);
        abort();
    }

    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_ecb(), NULL, (unsigned char *)common_aes_key, NULL) != 1) {
        ERR_print_errors_fp(stderr);
        abort();
    }
}

AESObject::~AESObject() {
    EVP_CIPHER_CTX_free(ctx);
}

__m128i AESObject::getRandomBlock()
{
    //generate more random seeds
	if (_mm_cvtsi128_si32(rCounter % RANDOM_COMPUTE == 0)) { 

        __m128i counter_buffer[RANDOM_COMPUTE];

		for (int i = 0; i < RANDOM_COMPUTE; i++) {
			counter_buffer[i] = rCounter + i;
        }
        
        int len = 0, ciphertext_len = 0;
        if(EVP_EncryptUpdate(
                ctx, 
                (unsigned char *) random_buffer,
                &len,
                (unsigned char *) counter_buffer,
                RANDOM_COMPUTE * BLOCK_BYTES
            ) != 1) {
            ERR_print_errors_fp(stderr);
            abort();
        }

        if(EVP_EncryptFinal_ex(ctx, ((unsigned char *) random_buffer) + len, &len) != 1) {
            ERR_print_errors_fp(stderr);
            abort();
        }
        ciphertext_len += len;

        assert(ciphertext_len == RANDOM_COMPUTE * BLOCK_BYTES);
	}

	return random_buffer[_mm_cvtsi128_si32(rCounter++ % RANDOM_COMPUTE)];
}

template<typename T>
void AESObject::getRandom(T *buf, int n) {

    assert(sizeof(T) < BLOCK_BYTES); // max 128 bit vals

    int blocks_required = ceil(
            ((double)(n * sizeof(T))) / BLOCK_BYTES
    );

    for (int i = 0; i < blocks_required; i++) {

        __m128i block = getRandomBlock();

        int buf_idx = i * BLOCK_BYTES;
        memcpy(((char *) buf) + buf_idx, &block, (n - buf_idx < BLOCK_BYTES) ? n : BLOCK_BYTES);
    }
}

