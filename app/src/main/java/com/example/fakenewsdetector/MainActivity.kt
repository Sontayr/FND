package com.example.fakenewsdetector

import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.style.TextDecoration
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                FakeNewsScreen()
            }
        }
    }
}

@Composable
fun FakeNewsScreen() {

    var text by remember { mutableStateOf(TextFieldValue("")) }

    var useMl by remember { mutableStateOf(true) }
    var useFactcheck by remember { mutableStateOf(true) }
    var useNews by remember { mutableStateOf(true) }
    var useLlm by remember { mutableStateOf(false) }

    var response by remember { mutableStateOf<PredictResponse?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }

    var showDetails by remember { mutableStateOf(false) }

    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {

        Text(
            text = "Fake News Detector",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        Text(
            text = "Проверка достоверности новостей с использованием ИИ",
            style = MaterialTheme.typography.bodyMedium
        )

        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Введите текст новости") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 8
        )

        Card {
            Column(
                modifier = Modifier.padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {

                Text(
                    text = "Методы проверки",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )

                SwitchRow("ML-модель", useMl) { useMl = it }
                SwitchRow("Google FactCheck", useFactcheck) { useFactcheck = it }
                SwitchRow("Новостные источники", useNews) { useNews = it }
                SwitchRow("LLM-анализ", useLlm) { useLlm = it }
            }
        }

        Button(
            enabled = !isLoading,
            onClick = {

                scope.launch {

                    isLoading = true
                    errorMessage = null
                    response = null

                    try {

                        response = ApiClient.api.predict(
                            PredictRequest(
                                text = text.text,
                                use_ml = useMl,
                                use_factcheck = useFactcheck,
                                use_news = useNews,
                                use_llm = useLlm
                            )
                        )

                    } catch (e: Exception) {
                        errorMessage = e.message
                    } finally {
                        isLoading = false
                    }
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Проверить")
        }

        if (isLoading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        errorMessage?.let {
            Text(
                text = "Ошибка: $it",
                color = MaterialTheme.colorScheme.error
            )
        }

        response?.let { resp ->

            val verdictText = when (resp.verdict) {
                "likely_true" -> "Скорее правда"
                "likely_false" -> "Скорее фейк"
                else -> "Недостаточно данных"
            }

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {

                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {

                    Text(
                        text = verdictText,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )

                    LinearProgressIndicator(
                        progress = { resp.truth_score.toFloat() },
                        modifier = Modifier.fillMaxWidth()
                    )

                    Text(
                        text = "Правдивость: ${(resp.truth_score * 100).toInt()}%"
                    )

                    if (resp.summary_points.isNotEmpty()) {

                        Text(
                            text = "Краткий анализ:",
                            fontWeight = FontWeight.SemiBold
                        )

                        resp.summary_points.forEach {
                            Text("• $it")
                        }
                    }

                    resp.llm_explanation?.let {

                        HorizontalDivider()

                        Text(
                            text = "LLM-анализ",
                            fontWeight = FontWeight.SemiBold
                        )

                        Text(it)
                    }

                    HorizontalDivider()

                    TextButton(
                        onClick = {
                            showDetails = !showDetails
                        }
                    ) {
                        Text(
                            if (showDetails)
                                "Скрыть технические детали"
                            else
                                "Показать технические детали"
                        )
                    }

                    AnimatedVisibility(showDetails) {

                        Column(
                            verticalArrangement = Arrangement.spacedBy(10.dp)
                        ) {

                            if (resp.signals.isNotEmpty()) {

                                Text(
                                    text = "Сигналы анализа",
                                    fontWeight = FontWeight.Bold
                                )

                                resp.signals.forEach {

                                    Card {

                                        Column(
                                            modifier = Modifier.padding(10.dp)
                                        ) {

                                            Text(
                                                text = signalDisplayName(it.name),
                                                fontWeight = FontWeight.SemiBold
                                            )

                                            Text(
                                                text = "Value: ${"%.2f".format(it.value)}"
                                            )

                                            Text(
                                                text = "Weight: ${"%.2f".format(it.weight)}"
                                            )

                                            it.detail?.let { detail ->
                                                Text(detail)
                                            }
                                        }
                                    }
                                }
                            }

                            if (resp.evidence.isNotEmpty()) {

                                Text(
                                    text = "Источники",
                                    fontWeight = FontWeight.Bold
                                )

                                val uriHandler = LocalUriHandler.current

                                resp.evidence.forEach { item ->

                                    Card {

                                        Column(
                                            modifier = Modifier.padding(10.dp)
                                        ) {

                                            Text(
                                                text = item.source,
                                                fontWeight = FontWeight.SemiBold
                                            )

                                            item.title?.let { title ->
                                                Spacer(modifier = Modifier.height(4.dp))
                                                Text(title)
                                            }

                                            item.score?.let { score ->
                                                Spacer(modifier = Modifier.height(4.dp))
                                                Text("Сходство: ${"%.2f".format(score)}")
                                            }

                                            item.note?.let { note ->
                                                Spacer(modifier = Modifier.height(4.dp))
                                                Text(
                                                    text = note,
                                                    style = MaterialTheme.typography.bodySmall
                                                )
                                            }

                                            item.url?.let { url ->
                                                Spacer(modifier = Modifier.height(4.dp))
                                                TextButton(
                                                    onClick = {
                                                        uriHandler.openUri(url)
                                                    }
                                                ) {
                                                    Text(
                                                        text = "Открыть источник",
                                                        textDecoration = TextDecoration.Underline
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if (resp.suspicious_fragments.isNotEmpty()) {

                                Text(
                                    text = "Подозрительные фрагменты",
                                    fontWeight = FontWeight.Bold
                                )

                                resp.suspicious_fragments.forEach {
                                    Text("• $it")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SwitchRow(
    label: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {

        Text(label)

        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange
        )
    }
}


fun signalDisplayName(name: String): String {
    return when (name) {
        "style_model" -> "ML-модель"
        "clickbait_heuristics" -> "Кликбейт-признаки"
        "factcheck" -> "Фактчекинг"
        "news_sources" -> "Новостные источники"
        "llm_analysis" -> "LLM-анализ"
        else -> name
    }
}