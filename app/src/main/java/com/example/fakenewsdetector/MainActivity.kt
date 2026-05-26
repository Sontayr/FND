package com.example.fakenewsdetector

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
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextDecoration
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

    var accessToken by remember { mutableStateOf<String?>(null) }
    var userEmail by remember { mutableStateOf<String?>(null) }

    var showLoginDialog by remember { mutableStateOf(false) }
    var showRegisterDialog by remember { mutableStateOf(false) }
    var showHistory by remember { mutableStateOf(false) }

    var historyItems by remember { mutableStateOf<List<HistoryItemResponse>>(emptyList()) }

    val scope = rememberCoroutineScope()

    if (showLoginDialog) {
        AuthDialog(
            title = "Вход в аккаунт",
            buttonText = "Войти",
            onDismiss = { showLoginDialog = false },
            onSubmit = { email, password ->
                scope.launch {
                    try {
                        val auth = ApiClient.api.login(AuthRequest(email, password))
                        accessToken = auth.access_token
                        userEmail = auth.email
                        showLoginDialog = false
                        errorMessage = null
                    } catch (e: Exception) {
                        errorMessage = "Ошибка входа: ${e.message}"
                    }
                }
            }
        )
    }

    if (showRegisterDialog) {
        AuthDialog(
            title = "Создать аккаунт",
            buttonText = "Зарегистрироваться",
            onDismiss = { showRegisterDialog = false },
            onSubmit = { email, password ->
                scope.launch {
                    try {
                        val auth = ApiClient.api.register(AuthRequest(email, password))
                        accessToken = auth.access_token
                        userEmail = auth.email
                        showRegisterDialog = false
                        errorMessage = null
                    } catch (e: Exception) {
                        errorMessage = "Ошибка регистрации: ${e.message}"
                    }
                }
            }
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {

        HeaderBlock(
            userEmail = userEmail,
            onLoginClick = { showLoginDialog = true },
            onRegisterClick = { showRegisterDialog = true },
            onLogoutClick = {
                accessToken = null
                userEmail = null
                historyItems = emptyList()
                showHistory = false
            },
            onHistoryClick = {
                scope.launch {
                    try {
                        val token = accessToken
                        if (token != null) {
                            historyItems = ApiClient.api.history("Bearer $token")
                            showHistory = true
                            errorMessage = null
                        }
                    } catch (e: Exception) {
                        errorMessage = "Ошибка загрузки истории: ${e.message}"
                    }
                }
            }
        )

        if (showHistory) {
            HistoryCard(
                items = historyItems,
                onClose = { showHistory = false }
            )
        }

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
                    showDetails = false

                    try {
                        val tokenHeader = accessToken?.let { "Bearer $it" }

                        response = ApiClient.api.predict(
                            req = PredictRequest(
                                text = text.text,
                                use_ml = useMl,
                                use_factcheck = useFactcheck,
                                use_news = useNews,
                                use_llm = useLlm
                            ),
                            authorization = tokenHeader
                        )

                    } catch (e: Exception) {
                        errorMessage = "Ошибка проверки: ${e.message}"
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
                text = it,
                color = MaterialTheme.colorScheme.error
            )
        }

        response?.let { resp ->
            ResultCard(
                resp = resp,
                showDetails = showDetails,
                onToggleDetails = { showDetails = !showDetails }
            )
        }
    }
}

@Composable
fun HeaderBlock(
    userEmail: String?,
    onLoginClick: () -> Unit,
    onRegisterClick: () -> Unit,
    onLogoutClick: () -> Unit,
    onHistoryClick: () -> Unit
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
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

        if (userEmail == null) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End
            ) {
                TextButton(onClick = onLoginClick) {
                    Text("Войти")
                }

                TextButton(onClick = onRegisterClick) {
                    Text("Создать аккаунт")
                }
            }
        } else {
            Column(
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "Вы вошли как: $userEmail",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    TextButton(onClick = onHistoryClick) {
                        Text("История")
                    }

                    TextButton(onClick = onLogoutClick) {
                        Text("Выйти")
                    }
                }
            }
        }
    }
}

@Composable
fun AuthDialog(
    title: String,
    buttonText: String,
    onDismiss: () -> Unit,
    onSubmit: (String, String) -> Unit
) {
    var email by remember { mutableStateOf(TextFieldValue("")) }
    var password by remember { mutableStateOf(TextFieldValue("")) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(title)
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text("Email") },
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("Пароль") },
                    visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    onSubmit(email.text.trim(), password.text.trim())
                }
            ) {
                Text(buttonText)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Отмена")
            }
        }
    )
}

@Composable
fun HistoryCard(
    items: List<HistoryItemResponse>,
    onClose: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "История проверок",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )

                TextButton(onClick = onClose) {
                    Text("Закрыть")
                }
            }

            if (items.isEmpty()) {
                Text("История пока пуста")
            } else {
                items.forEach { item ->
                    Card {
                        Column(
                            modifier = Modifier.padding(10.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = verdictDisplayName(item.verdict),
                                fontWeight = FontWeight.Bold
                            )

                            Text("Правдивость: ${(item.truth_score * 100).toInt()}%")

                            Text(
                                text = item.text_preview,
                                style = MaterialTheme.typography.bodyMedium
                            )

                            Text(
                                text = item.created_at,
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ResultCard(
    resp: PredictResponse,
    showDetails: Boolean,
    onToggleDetails: () -> Unit
) {
    val verdictText = verdictDisplayName(resp.verdict)

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
                onClick = onToggleDetails
            ) {
                Text(
                    if (showDetails)
                        "Скрыть технические детали"
                    else
                        "Показать технические детали"
                )
            }

            AnimatedVisibility(showDetails) {
                TechnicalDetails(resp)
            }
        }
    }
}

@Composable
fun TechnicalDetails(resp: PredictResponse) {
    Column(
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {

        if (resp.signals.isNotEmpty()) {
            Text(
                text = "Сигналы анализа",
                fontWeight = FontWeight.Bold
            )

            resp.signals.forEach { item ->
                Card {
                    Column(
                        modifier = Modifier.padding(10.dp)
                    ) {
                        Text(
                            text = signalDisplayName(item.name),
                            fontWeight = FontWeight.SemiBold
                        )

                        Text("Value: ${"%.2f".format(item.value)}")
                        Text("Weight: ${"%.2f".format(item.weight)}")

                        item.detail?.let { detail ->
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

fun verdictDisplayName(verdict: String): String {
    return when (verdict) {
        "likely_true" -> "Скорее правда"
        "likely_false" -> "Скорее фейк"
        else -> "Недостаточно данных"
    }
}