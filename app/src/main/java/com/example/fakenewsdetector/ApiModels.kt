package com.example.fakenewsdetector

data class PredictRequest(
    val text: String
)

data class SignalItem(
    val name: String,
    val value: Double,
    val weight: Double,
    val detail: String? = null
)

data class EvidenceItem(
    val source: String,
    val title: String? = null,
    val url: String? = null,
    val rating: String? = null,
    val score: Double? = null,
    val note: String? = null
)

data class PredictResponse(
    val label: String,
    val score: Double,
    val verdict: String,
    val truth_score: Double,
    val signals: List<SignalItem> = emptyList(),
    val suspicious_fragments: List<String> = emptyList(),
    val evidence: List<EvidenceItem> = emptyList()
)