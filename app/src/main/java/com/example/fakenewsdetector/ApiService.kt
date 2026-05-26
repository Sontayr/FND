package com.example.fakenewsdetector

import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Header
import retrofit2.http.POST

interface ApiService {

    @POST("predict")
    suspend fun predict(
        @Body req: PredictRequest,
        @Header("Authorization") authorization: String? = null
    ): PredictResponse

    @POST("auth/register")
    suspend fun register(
        @Body req: AuthRequest
    ): AuthResponse

    @POST("auth/login")
    suspend fun login(
        @Body req: AuthRequest
    ): AuthResponse

    @GET("auth/me")
    suspend fun me(
        @Header("Authorization") authorization: String
    ): UserResponse

    @GET("history")
    suspend fun history(
        @Header("Authorization") authorization: String
    ): List<HistoryItemResponse>
}