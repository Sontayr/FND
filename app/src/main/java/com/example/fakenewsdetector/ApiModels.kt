package com.example.fakenewsdetector

data class PredictRequest(
    val text: String,
    val use_ml: Boolean = true,
    val use_factcheck: Boolean = true,
    val use_news: Boolean = true,
    val use_llm: Boolean = false
)

data class PredictResponse(
    val label: String,
    val score: Double,
    val verdict: String,
    val truth_score: Double,
    val suspicious_fragments: List<String> = emptyList(),
    val evidence: List<EvidenceItem> = emptyList(),
    val signals: List<SignalItem> = emptyList(),
    val summary_points: List<String> = emptyList(),
    val llm_explanation: String? = null
)

data class EvidenceItem(
    val source: String,
    val title: String? = null,
    val url: String? = null,
    val rating: String? = null,
    val score: Double? = null,
    val note: String? = null
)

data class SignalItem(
    val name: String,
    val value: Double,
    val weight: Double,
    val detail: String? = null
)

data class AuthRequest(
    val email: String,
    val password: String
)

data class AuthResponse(
    val access_token: String,
    val token_type: String,
    val email: String
)

data class UserResponse(
    val id: Int,
    val email: String
)

data class HistoryItemResponse(
    val id: Int,
    val text_preview: String,
    val verdict: String,
    val truth_score: Double,
    val enabled_modules: String? = null,
    val created_at: String
)