package com.example.fakenewsdetector

import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Alignment
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
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
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp)
    ) {
        Column(
            modifier = Modifier
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color(0xFFEAF2FF),
                            Color(0xFFFFFFFF)
                        )
                    )
                )
                .padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Fake News Detector",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF163B73)
            )

            Text(
                text = "Интеллектуальная проверка достоверности новостей по нескольким источникам: ML-модель, фактчекинг, СМИ и LLM-анализ.",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF42526E)
            )

            if (userEmail == null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(onClick = onLoginClick) {
                        Text("Войти")
                    }

                    Button(
                        onClick = onRegisterClick,
                        shape = RoundedCornerShape(14.dp)
                    ) {
                        Text("Создать аккаунт")
                    }
                }
            } else {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(18.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFFF4F8FF)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "Профиль",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF163B73)
                        )

                        Text(
                            text = userEmail,
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color(0xFF42526E)
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
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(22.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
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
                Text(
                    text = "История пока пуста",
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                items.forEach { item ->
                    val color = verdictColor(item.verdict)
                    val dateText = formatHistoryDate(item.created_at)

                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(18.dp),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFFF8F9FC)
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(14.dp),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text(
                                text = verdictDisplayName(item.verdict),
                                color = color,
                                fontWeight = FontWeight.Bold,
                                style = MaterialTheme.typography.titleMedium
                            )

                            Text(
                                text = dateText,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            Text(
                                text = item.text_preview,
                                style = MaterialTheme.typography.bodyMedium
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
    val color = verdictColor(resp.verdict)
    val bgColor = verdictBackgroundColor(resp.verdict)

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(24.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp)
    ) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Card(
                shape = RoundedCornerShape(18.dp),
                colors = CardDefaults.cardColors(containerColor = bgColor)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text = verdictText,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = color
                    )

                    LinearProgressIndicator(
                        progress = { resp.truth_score.toFloat().coerceIn(0f, 1f) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(10.dp),
                        color = color
                    )

                    Text(
                        text = "Оценка правдивости: ${(resp.truth_score * 100).toInt()}%",
                        fontWeight = FontWeight.SemiBold,
                        color = color
                    )
                }
            }

            if (resp.summary_points.isNotEmpty()) {
                Text(
                    text = "Краткий анализ",
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.titleMedium
                )

                resp.summary_points.forEach {
                    Text(
                        text = "• $it",
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            resp.llm_explanation?.let {
                HorizontalDivider()

                Text(
                    text = "LLM-анализ",
                    fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.titleMedium
                )

                Text(
                    text = it,
                    style = MaterialTheme.typography.bodyMedium
                )
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
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        if (resp.signals.isNotEmpty()) {
            Text(
                text = "Подробности проверки",
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium
            )

            resp.signals.forEach { item ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFFF8F9FC)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            text = signalDisplayName(item.name),
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = signalHumanText(item),
                            style = MaterialTheme.typography.bodyMedium
                        )

                        item.detail?.let { detail ->
                            Text(
                                text = detail,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }
            }
        }

        if (resp.evidence.isNotEmpty()) {
            Text(
                text = "Найденные источники",
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium
            )

            val uriHandler = LocalUriHandler.current

            resp.evidence.forEach { item ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFFF8F9FC)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(
                            text = item.source,
                            fontWeight = FontWeight.Bold
                        )

                        item.title?.let { title ->
                            Text(title)
                        }

                        item.score?.let { score ->
                            Text(
                                text = sourceMatchText(score),
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        item.note?.let { note ->
                            Text(
                                text = note,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        item.url?.let { url ->
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
                fontWeight = FontWeight.Bold,
                style = MaterialTheme.typography.titleMedium
            )

            resp.suspicious_fragments.forEach {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFFFFF8E6)
                    )
                ) {
                    Text(
                        text = it,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodyMedium
                    )
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

fun verdictDisplayName(verdict: String): String {
    return when (verdict) {
        "likely_true" -> "Скорее правда"
        "likely_false" -> "Скорее фейк"
        else -> "Недостаточно данных"
    }
}


fun verdictColor(verdict: String): Color {
    return when (verdict) {
        "likely_true" -> Color(0xFF1B7F3A)
        "likely_false" -> Color(0xFFC62828)
        else -> Color(0xFF5E35B1)
    }
}

fun verdictBackgroundColor(verdict: String): Color {
    return when (verdict) {
        "likely_true" -> Color(0xFFE8F5E9)
        "likely_false" -> Color(0xFFFFEBEE)
        else -> Color(0xFFF3E5F5)
    }
}

fun signalHumanText(signal: SignalItem): String {
    return when {
        signal.name == "style_model" && signal.value >= 0.65 ->
            "Стиль текста выглядит достаточно надежным по оценке ML-модели."

        signal.name == "style_model" && signal.value <= 0.35 ->
            "ML-модель обнаружила признаки недостоверного или подозрительного стиля."

        signal.name == "clickbait_heuristics" && signal.value >= 0.65 ->
            "Выраженных кликбейт-признаков не обнаружено."

        signal.name == "clickbait_heuristics" && signal.value <= 0.35 ->
            "Обнаружены признаки эмоционального или кликбейтного оформления."

        signal.name == "factcheck" && signal.value >= 0.65 ->
            "Фактчекинговые источники скорее подтверждают информацию."

        signal.name == "factcheck" && signal.value <= 0.35 ->
            "Фактчекинговые источники указывают на возможное опровержение."

        signal.name == "factcheck" ->
            "Фактчекинг не дал уверенного результата."

        signal.name == "news_sources" && signal.value >= 0.65 ->
            "Похожие материалы найдены в новостных источниках."

        signal.name == "news_sources" ->
            "Новостные источники не дали уверенного подтверждения."

        signal.name == "llm_analysis" && signal.value >= 0.65 ->
            "LLM-анализ не выявил явных противоречий с общеизвестными фактами."

        signal.name == "llm_analysis" && signal.value <= 0.35 ->
            "LLM-анализ обнаружил возможные фактические или логические противоречия."

        signal.name == "llm_analysis" ->
            "LLM-анализ не дал уверенного вывода."

        else ->
            "Сигнал обработан системой, но не дал однозначного вывода."
    }
}

fun sourceMatchText(score: Double): String {
    return when {
        score >= 0.35 -> "Найдено сильное сходство с источником"
        score >= 0.20 -> "Найдено частичное сходство с источником"
        else -> "Найдено слабое сходство с источником"
    }
}

fun formatHistoryDate(raw: String): String {
    return try {
        raw
            .replace("T", " ")
            .substringBefore(".")
    } catch (e: Exception) {
        raw
    }
}