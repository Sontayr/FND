package com.example.fakenewsdetector

data class PredictRequest(val text: String)
data class PredictResponse(val label: String, val score: Double)