package com.example.fakenewsdetector

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    FakeNewsScreen()
                }
            }
        }
    }
}

@Composable
fun FakeNewsScreen() {
    var text by remember { mutableStateOf(TextFieldValue("")) }
    var response by remember { mutableStateOf<PredictResponse?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }

    val scope = rememberCoroutineScope()
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        Text(
            text = "Проверка новости",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Вставьте текст новости, и приложение оценит ее достоверность по нескольким сигналам.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    label = { Text("Текст новости") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 8,
                    maxLines = 14
                )

                Spacer(modifier = Modifier.height(10.dp))

                Row(modifier = Modifier.fillMaxWidth()) {
                    Button(
                        onClick = {
                            scope.launch {
                                val input = text.text.trim()
                                if (input.isEmpty()) {
                                    errorMessage = "Введите текст новости"
                                    response = null
                                    return@launch
                                }

                                isLoading = true
                                errorMessage = null
                                response = null

                                try {
                                    val resp = ApiClient.api.predict(PredictRequest(input))
                                    response = resp
                                } catch (e: Exception) {
                                    errorMessage = "Ошибка запроса: ${e.message}"
                                } finally {
                                    isLoading = false
                                }
                            }
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Проверить")
                    }

                    Spacer(modifier = Modifier.width(10.dp))

                    TextButton(
                        onClick = {
                            text = TextFieldValue("")
                            response = null
                            errorMessage = null
                        },
                        modifier = Modifier.align(Alignment.CenterVertically)
                    ) {
                        Text("Очистить")
                    }
                }
            }
        }

        if (isLoading) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 3.dp)
            ) {
                Row(
                    modifier = Modifier.padding(18.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator()
                    Spacer(modifier = Modifier.width(16.dp))
                    Column {
                        Text(
                            text = "Идет анализ",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "Модель и источники проверяют текст...",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }

        errorMessage?.let { msg ->
            StatusMessageCard(
                title = "Ошибка",
                text = msg,
                backgroundColor = Color(0xFFFFF0F0)
            )
        }

        response?.let { resp ->
            VerdictCard(resp)
            SuspiciousFragmentsCard(resp)
            SignalsCard(resp)
            EvidenceCard(resp)
        }
    }
}

@Composable
fun VerdictCard(resp: PredictResponse) {
    val verdictText = when (resp.verdict) {
        "likely_true" -> "Скорее правда"
        "likely_false" -> "Скорее фейк"
        else -> "Недостаточно данных"
    }

    val verdictDescription = when (resp.verdict) {
        "likely_true" -> "Текст выглядит относительно достоверным по совокупности проверок."
        "likely_false" -> "Текст содержит признаки недостоверной или сомнительной информации."
        else -> "Система не нашла достаточно оснований для уверенного вывода."
    }

    val badgeColor = when (resp.verdict) {
        "likely_true" -> Color(0xFFE7F6EC)
        "likely_false" -> Color(0xFFFFECEC)
        else -> Color(0xFFF3F0FF)
    }

    val badgeTextColor = when (resp.verdict) {
        "likely_true" -> Color(0xFF1F7A3D)
        "likely_false" -> Color(0xFFB3261E)
        else -> Color(0xFF5B4BA8)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 5.dp)
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = "Результат проверки",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(14.dp))

            Box(
                modifier = Modifier
                    .background(badgeColor, RoundedCornerShape(14.dp))
                    .padding(horizontal = 14.dp, vertical = 10.dp)
            ) {
                Text(
                    text = verdictText,
                    color = badgeTextColor,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
            }

            Spacer(modifier = Modifier.height(14.dp))

            Text(
                text = verdictDescription,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(16.dp))

            ScoreBar(
                label = "Вероятность правдивости",
                value = resp.truth_score
            )

            Spacer(modifier = Modifier.height(14.dp))

            InfoRow("Итоговая оценка", formatPercent(resp.truth_score))
            InfoRow("Технический score", String.format("%.2f", resp.score))
            InfoRow(
                "Метка модели",
                when (resp.label) {
                    "real" -> "Правда"
                    "fake" -> "Фейк"
                    else -> "Не определено"
                }
            )
        }
    }
}

@Composable
fun SuspiciousFragmentsCard(resp: PredictResponse) {
    if (resp.suspicious_fragments.isEmpty()) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp)
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = "Подозрительные фрагменты",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(10.dp))

            resp.suspicious_fragments.forEachIndexed { index, fragment ->
                FragmentItem(index + 1, fragment)
                if (index != resp.suspicious_fragments.lastIndex) {
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }
        }
    }
}

@Composable
fun SignalsCard(resp: PredictResponse) {
    if (resp.signals.isEmpty()) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp)
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = "Сигналы анализа",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(10.dp))

            resp.signals.forEachIndexed { index, signal ->
                val title = when (signal.name) {
                    "style_model" -> "Модель текста"
                    "clickbait_heuristics" -> "Кликбейт-признаки"
                    "factcheck" -> "Фактчекинг"
                    "news_sources" -> "Новостные источники"
                    else -> signal.name
                }

                SignalItemCard(
                    title = title,
                    value = signal.value,
                    weight = signal.weight,
                    detail = signal.detail
                )

                if (index != resp.signals.lastIndex) {
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }
        }
    }
}

@Composable
fun EvidenceCard(resp: PredictResponse) {
    if (resp.evidence.isEmpty()) return

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp)
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text(
                text = "Найденные источники",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(10.dp))

            resp.evidence.forEachIndexed { index, item ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
                    ),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Text(
                            text = item.source,
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold
                        )

                        item.title?.takeIf { it.isNotBlank() }?.let {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(text = it, style = MaterialTheme.typography.bodyMedium)
                        }

                        item.note?.takeIf { it.isNotBlank() }?.let {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = it,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        item.rating?.takeIf { it.isNotBlank() }?.let {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(text = "Оценка источника: $it", style = MaterialTheme.typography.bodySmall)
                        }

                        item.score?.let {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Сходство: ${String.format("%.2f", it)}",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }

                        item.url?.takeIf { it.isNotBlank() }?.let {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = it,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.primary
                            )
                        }
                    }
                }

                if (index != resp.evidence.lastIndex) {
                    Spacer(modifier = Modifier.height(10.dp))
                }
            }
        }
    }
}

@Composable
fun SignalItemCard(
    title: String,
    value: Double,
    weight: Double,
    detail: String?
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold
            )

            Spacer(modifier = Modifier.height(8.dp))
            ScoreBar(label = "Вклад сигнала", value = value)

            Spacer(modifier = Modifier.height(8.dp))
            InfoRow("Вес", String.format("%.2f", weight))

            detail?.takeIf { it.isNotBlank() }?.let {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun FragmentItem(index: Int, text: String) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(modifier = Modifier.padding(14.dp)) {
            Box(
                modifier = Modifier
                    .background(
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.12f),
                        shape = RoundedCornerShape(12.dp)
                    )
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Text(
                    text = index.toString(),
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            Spacer(modifier = Modifier.width(12.dp))

            Text(
                text = text,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
fun ScoreBar(label: String, value: Double) {
    val clamped = value.coerceIn(0.0, 1.0).toFloat()

    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(label, style = MaterialTheme.typography.bodyMedium)
            Text(formatPercent(value), fontWeight = FontWeight.SemiBold)
        }

        Spacer(modifier = Modifier.height(8.dp))

        androidx.compose.material3.LinearProgressIndicator(
            progress = { clamped },
            modifier = Modifier
                .fillMaxWidth()
                .height(10.dp),
        )
    }
}

@Composable
fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
fun StatusMessageCard(
    title: String,
    text: String,
    backgroundColor: Color
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = backgroundColor)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(text = text, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

fun formatPercent(value: Double): String {
    return String.format("%.1f%%", value.coerceIn(0.0, 1.0) * 100)
}