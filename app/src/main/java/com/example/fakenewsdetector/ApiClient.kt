package com.example.fakenewsdetector

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object ApiClient {
    // Раз у тебя сейчас это работает в эмуляторе, пока оставляем как есть.
    // Если потом перестанет работать в эмуляторе, поменяешь на http://10.0.2.2:8000/
    //private const val BASE_URL = "http://127.0.0.1:8000/"
    private const val BASE_URL = "http://192.168.2.140:8000/"

    val api: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}