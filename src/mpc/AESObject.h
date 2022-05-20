
#pragma once

#include <algorithm>
#include <emmintrin.h>
#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>

#include "../globals.h"

#define BLOCK_BYTES 16

class AESObject {

    private:

        EVP_CIPHER_CTX *ctx;

        __m128i rCounter = _mm_setzero_si128();
        __m128i random_buffer[RANDOM_COMPUTE];

        __m128i getRandomBlock();

    public:

        AESObject(char* filename);
        ~AESObject(); 

        template<typename T> void getRandom(T *buf, int n);
};

