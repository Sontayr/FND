package com.example.fakenewsdetector

import retrofit2.http.Body
import retrofit2.http.POST

interface ApiService {
    @POST("predict")
    suspend fun predict(@Body req: PredictRequest): PredictResponse
}