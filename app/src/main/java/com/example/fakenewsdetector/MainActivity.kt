package com.example.fakenewsdetector

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
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
    var result by remember { mutableStateOf("Результат появится здесь") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("Проверка новости", style = MaterialTheme.typography.headlineSmall)

        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Вставьте текст новости") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 6
        )

        val scope = rememberCoroutineScope()

        Button(
            onClick = {
                scope.launch {
                    result = "Проверяю..."
                    try {
                        val resp = ApiClient.api.predict(PredictRequest(text.text))
                        val pretty = if (resp.label == "fake") "Фейк" else "Правда"
                        result = "$pretty (score=${"%.2f".format(resp.score)})"
                    } catch (e: Exception) {
                        result = "Ошибка: ${e.message}"
                    }
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Проверить")
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = result,
                modifier = Modifier.padding(16.dp),
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}