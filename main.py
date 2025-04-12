import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from tkinterweb import HtmlFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import logging
import os
import time
import requests
import webbrowser

# Configuração de logging
logging.basicConfig(
    filename='football_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)

class DataCollector:
    """Classe responsável pela coleta de dados da API-Futebol."""
    def __init__(self, api_key, campeonato_id=14):
        self.api_key = api_key
        self.base_url = "https://api.api-futebol.com.br/v1/campeonatos"
        self.campeonato_id = campeonato_id
        self.cache_file = f'data_cache_{campeonato_id}.pkl'
        self.cache_duration = 3600  # 1 hora

    def fetch_campeonatos(self):
        """Obtém a lista de campeonatos disponíveis."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = self.base_url
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return {campeonato['nome']: campeonato['campeonato_id'] for campeonato in data}
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao obter lista de campeonatos: {str(e)}")
            return {}

    def fetch_data(self, force_update=False):
        """Faz a requisição à API-Futebol para obter a tabela de classificação."""
        if not force_update and self._check_cache():
            return self._load_cache()

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/{self.campeonato_id}/tabela"
        try:
            logging.info(f"Acessando URL: {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            logging.info("Sucesso ao carregar dados da API")
            if not data:
                raise ValueError("A API retornou uma resposta vazia")
            df = self._parse_data_to_df(data)
            self._save_cache(df)
            return df
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao acessar a API: {str(e)}")
            return None

    def get_data(self, force_update=False) -> pd.DataFrame:
        """Coleta dados da API-Futebol."""
        data = self.fetch_data(force_update)
        if data is not None:
            return data
        raise Exception("Falha na coleta: Não foi possível carregar os dados")

    def _parse_data_to_df(self, data: list) -> pd.DataFrame:
        """Extrai dados da resposta da API e retorna um DataFrame."""
        dados = []
        for item in data:
            try:
                time = item.get("time", {})
                nome_time = time.get("nome_popular", "")
                if not nome_time:
                    logging.warning("Nome do time não encontrado na resposta")
                    continue
                dados.append({
                    "Time": nome_time,
                    "Jogos": item.get("jogos", 0),
                    "Vitorias": item.get("vitorias", 0),
                    "Empates": item.get("empates", 0),
                    "Derrotas": item.get("derrotas", 0),
                    "Gols_Pro": item.get("gols_pro", 0),
                    "Gols_Contra": item.get("gols_contra", 0),
                    "Pontos": item.get("pontos", 0)
                })
            except Exception as e:
                logging.warning(f"Erro ao extrair dados do time: {str(e)}")
        if not dados:
            raise ValueError("Não foi possível extrair dados da API")
        df = pd.DataFrame(dados)
        required_columns = ['Time', 'Jogos', 'Vitorias', 'Empates', 'Derrotas', 'Gols_Pro', 'Gols_Contra', 'Pontos']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Colunas ausentes no DataFrame: {missing_cols}")
        if df.empty:
            raise ValueError("O DataFrame está vazio após o parsing")
        logging.info(f"Dados coletados com sucesso. Shape: {df.shape}")
        return df

    def _check_cache(self) -> bool:
        """Verifica se o cache é válido."""
        if not os.path.exists(self.cache_file):
            return False
        return time.time() - os.path.getmtime(self.cache_file) < self.cache_duration

    def _load_cache(self) -> pd.DataFrame:
        """Carrega dados do cache."""
        return pd.read_pickle(self.cache_file)

    def _save_cache(self, df: pd.DataFrame):
        """Salva dados no cache."""
        df.to_pickle(self.cache_file)

class DataProcessor:
    """Classe para processar os dados coletados."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()

    def process_data(self) -> pd.DataFrame:
        """Processa os dados e calcula métricas."""
        required_columns = ['Time', 'Jogos', 'Vitorias', 'Empates', 'Derrotas', 'Gols_Pro', 'Gols_Contra', 'Pontos']
        if not all(col in self.df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in self.df.columns]
            raise ValueError(f"Dados incompletos ou inválidos. Colunas ausentes: {missing_cols}")

        if self.df.empty:
            raise ValueError("O DataFrame está vazio")

        self.df['Saldo_Gols'] = self.df['Gols_Pro'] - self.df['Gols_Contra']
        self.df['Aproveitamento'] = self.df.apply(
            lambda row: (row['Pontos'] / (row['Jogos'] * 3)) * 100 if row['Jogos'] > 0 else 0, axis=1
        )
        self.df['Media_Gols_Pro'] = self.df.apply(
            lambda row: row['Gols_Pro'] / row['Jogos'] if row['Jogos'] > 0 else 0, axis=1
        )
        self.df['Media_Gols_Contra'] = self.df.apply(
            lambda row: row['Gols_Contra'] / row['Jogos'] if row['Jogos'] > 0 else 0, axis=1
        )

        if self.df[['Aproveitamento', 'Saldo_Gols', 'Media_Gols_Pro']].isnull().any().any():
            raise ValueError("Valores NaN encontrados após o cálculo das métricas")

        features_scaled = self.scaler.fit_transform(self.df[['Aproveitamento', 'Saldo_Gols', 'Media_Gols_Pro']])
        weights = [0.5, 0.3, 0.2]
        self.df['Score'] = np.sum(features_scaled * weights, axis=1)

        logging.info(f"Dados processados com sucesso. Shape: {self.df.shape}")
        return self.df.sort_values('Score', ascending=False)

class PredictionModel:
    """Classe para prever resultados de partidas."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def train_model(self):
        """Treina o modelo de previsão."""
        features = ['Jogos', 'Vitorias', 'Empates', 'Derrotas', 'Saldo_Gols', 'Media_Gols_Pro', 'Media_Gols_Contra']
        if not all(col in self.df.columns for col in features):
            missing_cols = [col for col in features if col not in self.df.columns]
            raise ValueError(f"Colunas de features ausentes no DataFrame: {missing_cols}")

        if 'Aproveitamento' not in self.df.columns:
            raise ValueError("Coluna 'Aproveitamento' não encontrada no DataFrame")

        y = (self.df['Aproveitamento'] > self.df['Aproveitamento'].mean()).astype(int)
        if y.empty:
            raise ValueError("O vetor de rótulos 'y' está vazio")

        X = self.df[features]
        if X.empty:
            raise ValueError("O DataFrame de features 'X' está vazio")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        logging.info(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def predict_match(self, time1: str, time2: str) -> dict:
        """Prevê o resultado entre dois times com mais detalhes."""
        features = ['Jogos', 'Vitorias', 'Empates', 'Derrotas', 'Saldo_Gols', 'Media_Gols_Pro', 'Media_Gols_Contra']
        time1_data = self.df[self.df['Time'] == time1][features]
        time2_data = self.df[self.df['Time'] == time2][features]
        if time1_data.empty or time2_data.empty:
            raise ValueError("Time não encontrado na base de dados")

        prob_time1 = self.model.predict_proba(time1_data)[0][1] * 100
        prob_time2 = self.model.predict_proba(time2_data)[0][1] * 100

        all_probs = self.model.predict_proba(self.df[features])[:, 1] * 100
        std_prob = np.std(all_probs)
        confidence_interval = 1.96 * std_prob / np.sqrt(len(all_probs))

        odds_time1 = 100 / prob_time1 if prob_time1 > 0 else float('inf')
        odds_time2 = 100 / prob_time2 if prob_time2 > 0 else float('inf')

        time1_stats = self.df[self.df['Time'] == time1].iloc[0]
        time2_stats = self.df[self.df['Time'] == time2].iloc[0]
        avg_goals = (time1_stats['Media_Gols_Pro'] + time1_stats['Media_Gols_Contra'] + 
                     time2_stats['Media_Gols_Pro'] + time2_stats['Media_Gols_Contra']) / 2
        prob_over_2_5 = (avg_goals - 2.5) / avg_goals if avg_goals > 0 else 0
        prob_over_2_5 = max(0, min(100, prob_over_2_5 * 100))
        prob_under_2_5 = 100 - prob_over_2_5
        prob_ambos_marcam = (time1_stats['Media_Gols_Pro'] * time2_stats['Media_Gols_Pro']) / (avg_goals ** 2) * 100 if avg_goals > 0 else 0
        prob_ambos_marcam = max(0, min(100, prob_ambos_marcam))

        return {
            'time1': {
                'nome': time1,
                'probabilidade': prob_time1,
                'intervalo_confianca': confidence_interval,
                'aproveitamento': time1_stats['Aproveitamento'],
                'saldo_gols': time1_stats['Saldo_Gols'],
                'media_gols_pro': time1_stats['Media_Gols_Pro'],
                'media_gols_contra': time1_stats['Media_Gols_Contra'],
                'odds': odds_time1
            },
            'time2': {
                'nome': time2,
                'probabilidade': prob_time2,
                'intervalo_confianca': confidence_interval,
                'aproveitamento': time2_stats['Aproveitamento'],
                'saldo_gols': time2_stats['Saldo_Gols'],
                'media_gols_pro': time2_stats['Media_Gols_Pro'],
                'media_gols_contra': time2_stats['Media_Gols_Contra'],
                'odds': odds_time2
            },
            'mercados': {
                'mais_2_5_gols': prob_over_2_5,
                'menos_2_5_gols': prob_under_2_5,
                'ambos_marcam': prob_ambos_marcam
            },
            'sugestao': time1 if prob_time1 > prob_time2 else time2
        }

class FootballAnalysisGUI:
    """Interface gráfica para análise de futebol."""
    def __init__(self, api_key):
        self.api_key = api_key
        self.root = tk.Tk()
        self.root.title("BetMaster Pro - Análise Avançada de Apostas")
        self.root.geometry("1366x768")
        self.root.configure(bg='#121212')

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(".", background="#121212", foreground="white", fieldbackground="#1e1e1e")
        self.style.configure("TNotebook", background="#121212", foreground="white")
        self.style.configure("TNotebook.Tab", background="#1e1e1e", foreground="white", padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", "#00cc66")], foreground=[("selected", "white")])
        self.style.configure("TButton", background="#00cc66", foreground="white", padding=8, font=("Arial", 11, "bold"))
        self.style.map("TButton", background=[('active', '#00b359')])
        self.style.configure("TCombobox", fieldbackground="#1e1e1e", background="#1e1e1e", foreground="white")
        self.style.configure("Treeview", background="#1e1e1e", foreground="white", fieldbackground="#1e1e1e")
        self.style.configure("Treeview.Heading", background="#00cc66", foreground="white", font=("Arial", 11, "bold"))

        self.collector = DataCollector(api_key=self.api_key)
        self.campeonatos = self.collector.fetch_campeonatos()
        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """Configura a interface gráfica."""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        menu_frame = ttk.Frame(main_frame, style="TFrame")
        menu_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(menu_frame, text="Campeonato:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.combo_campeonato = ttk.Combobox(menu_frame, values=list(self.campeonatos.keys()), state='readonly', font=("Arial", 11))
        self.combo_campeonato.pack(side=tk.LEFT, padx=5)
        self.combo_campeonato.bind("<<ComboboxSelected>>", self._change_campeonato)
        self.combo_campeonato.set("Brasileirão Série B")

        self.update_button = ttk.Button(menu_frame, text="Atualizar Dados", command=self._update_data_with_feedback)
        self.update_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(menu_frame, text="", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.tab_control = ttk.Notebook(main_frame)
        self.tab_dashboard = ttk.Frame(self.tab_control)
        self.tab_analise = ttk.Frame(self.tab_control)
        self.tab_previsoes = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_dashboard, text='Dashboard')
        self.tab_control.add(self.tab_analise, text='Análise Geral')
        self.tab_control.add(self.tab_previsoes, text='Previsão de Partidas')
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        self.tab_control.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self._setup_dashboard_tab()
        self._setup_analise_tab()
        self._setup_previsoes_tab()

    def _setup_dashboard_tab(self):
        """Configura a aba de dashboard usando widgets Tkinter."""
        frame_dashboard = ttk.Frame(self.tab_dashboard, padding=10)
        frame_dashboard.pack(fill=tk.BOTH, expand=True)

        frame_top5 = ttk.LabelFrame(frame_dashboard, text="Top 5 Times por Score", padding=10)
        frame_top5.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.top5_frame = ttk.Frame(frame_top5)
        self.top5_frame.pack(fill=tk.BOTH, expand=True)

        frame_forma = ttk.LabelFrame(frame_dashboard, text="Times em Melhor Forma", padding=10)
        frame_forma.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.forma_frame = ttk.Frame(frame_forma)
        self.forma_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_analise_tab(self):
        """Configura a aba de análise geral."""
        frame_tabela = ttk.LabelFrame(self.tab_analise, text="Tabela de Classificação", padding=10)
        frame_tabela.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        colunas = ('Time', 'Pontos', 'Vitórias', 'Aproveitamento', 'Score')
        self.tabela = ttk.Treeview(frame_tabela, columns=colunas, show='headings')
        for col in colunas:
            self.tabela.heading(col, text=col)
            self.tabela.column(col, width=120)
        scrollbar = ttk.Scrollbar(frame_tabela, orient=tk.VERTICAL, command=self.tabela.yview)
        self.tabela.configure(yscrollcommand=scrollbar.set)
        self.tabela.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        frame_graficos = ttk.LabelFrame(self.tab_analise, text="Gráficos", padding=10)
        frame_graficos.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.grafico_analise_html = HtmlFrame(frame_graficos)
        self.grafico_analise_html.pack(fill=tk.BOTH, expand=True)

    def _setup_previsoes_tab(self):
        """Configura a aba de previsões."""
        frame_previsao = ttk.LabelFrame(self.tab_previsoes, text="Previsão de Partidas", padding=10)
        frame_previsao.pack(fill=tk.BOTH, expand=True)

        frame_times = ttk.Frame(frame_previsao)
        frame_times.pack(fill=tk.X, pady=10)
        ttk.Label(frame_times, text="Time 1:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.combo_time1 = ttk.Combobox(frame_times, state='readonly', font=("Arial", 11))
        self.combo_time1.pack(side=tk.LEFT, padx=5)
        ttk.Label(frame_times, text="Time 2:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.combo_time2 = ttk.Combobox(frame_times, state='readonly', font=("Arial", 11))
        self.combo_time2.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame_previsao, text="Realizar Previsão", command=self._fazer_previsao).pack(pady=10)

        self.resultado_text = tk.Text(frame_previsao, height=20, width=60, bg='#1e1e1e', fg='white', 
                                     font=("Arial", 12), borderwidth=2, relief="groove")
        self.resultado_text.pack(pady=10)

        # Configurar tags para estilização
        self.resultado_text.tag_configure("header", font=("Arial", 14, "bold"), foreground="#00cc66")
        self.resultado_text.tag_configure("subheader", font=("Arial", 12, "bold"), foreground="#ffcc00")
        self.resultado_text.tag_configure("time1", foreground="#00cc66")
        self.resultado_text.tag_configure("time2", foreground="#ff4444")
        self.resultado_text.tag_configure("normal", foreground="white")

        frame_grafico = ttk.Frame(frame_previsao)
        frame_grafico.pack(fill=tk.BOTH, expand=True)
        self.grafico_previsao_html = HtmlFrame(frame_grafico)
        self.grafico_previsao_html.pack(fill=tk.BOTH, expand=True)

    def _change_campeonato(self, event):
        """Atualiza o campeonato selecionado."""
        campeonato_nome = self.combo_campeonato.get()
        self.collector.campeonato_id = self.campeonatos[campeonato_nome]
        self.collector.cache_file = f'data_cache_{self.collector.campeonato_id}.pkl'
        self._update_data_with_feedback()

    def _update_data_with_feedback(self):
        """Atualiza os dados com feedback visual."""
        self.status_label.config(text="Carregando dados...", foreground="#ffcc00")
        self.update_button.state(['disabled'])
        self.root.update()
        self.root.after(100, self._load_data)

    def _load_data(self):
        """Carrega e processa os dados."""
        try:
            self.df = self.collector.get_data(force_update=True)
            logging.info(f"DataFrame após coleta: {self.df}")
            processor = DataProcessor(self.df)
            self.df_processed = processor.process_data()
            logging.info(f"DataFrame após processamento: {self.df_processed}")
            self.prediction_model = PredictionModel(self.df_processed)
            self.prediction_model.train_model()
            self._update_ui()
            self.status_label.config(text="Dados atualizados com sucesso!", foreground="#00cc66")
        except Exception as e:
            self.status_label.config(text=f"Erro ao atualizar dados: {str(e)}", foreground="#ff4444")
            messagebox.showerror("Erro", f"Erro ao carregar dados: {str(e)}\nTente novamente mais tarde.")
        finally:
            self.update_button.state(['!disabled'])

    def _on_tab_change(self, event):
        """Gerencia a mudança de abas para evitar travamentos."""
        self.status_label.config(text="Carregando aba...", foreground="#ffcc00")
        self.root.update()
        self.root.after(100, self._update_current_tab)

    def _update_current_tab(self):
        """Atualiza a aba atual para evitar travamentos."""
        self._update_ui()
        self.status_label.config(text="", foreground="white")

    def _update_ui(self):
        """Atualiza a interface com os dados processados."""
        # Atualiza a aba de dashboard
        for widget in self.top5_frame.winfo_children():
            widget.destroy()
        top5 = self.df_processed.head(5)
        ttk.Label(self.top5_frame, text="Top 5 Times", font=("Arial", 14, "bold"), foreground="#00cc66").pack(anchor="w")
        for _, row in top5.iterrows():
            ttk.Label(
                self.top5_frame,
                text=f"{row['Time']}: Score {row['Score']:.2f}, Aproveitamento {row['Aproveitamento']:.1f}%",
                font=("Arial", 12),
                foreground="white"
            ).pack(anchor="w", pady=2)

        for widget in self.forma_frame.winfo_children():
            widget.destroy()
        forma = self.df_processed.sort_values('Aproveitamento', ascending=False).head(5)
        ttk.Label(self.forma_frame, text="Times em Melhor Forma", font=("Arial", 14, "bold"), foreground="#00cc66").pack(anchor="w")
        for _, row in forma.iterrows():
            ttk.Label(
                self.forma_frame,
                text=f"{row['Time']}: Aproveitamento {row['Aproveitamento']:.1f}%",
                font=("Arial", 12),
                foreground="white"
            ).pack(anchor="w", pady=2)

        # Atualiza a aba de análise geral
        for item in self.tabela.get_children():
            self.tabela.delete(item)
        for _, row in self.df_processed.iterrows():
            score = row['Score']
            tag = 'alto' if score > 1 else 'baixo' if score < -1 else 'medio'
            self.tabela.insert("", tk.END, values=(
                row['Time'], row['Pontos'], row['Vitorias'], f"{row['Aproveitamento']:.1f}%", f"{row['Score']:.2f}"
            ), tags=(tag,))
        self.tabela.tag_configure('alto', background='#00cc66')
        self.tabela.tag_configure('medio', background='#ffcc00')
        self.tabela.tag_configure('baixo', background='#ff4444')

        # Gráfico de análise geral (interativo com Plotly)
        top_teams = self.df_processed.head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_teams['Time'],
            y=top_teams['Score'],
            marker_color='#00cc66',
            text=[f"{score:.2f}" for score in top_teams['Score']],
            textposition='auto'
        ))
        fig.update_layout(
            title="Top 10 Times por Score",
            title_font=dict(size=20, color='white'),
            paper_bgcolor='#121212',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            xaxis_title="Times",
            yaxis_title="Score",
            xaxis=dict(tickangle=45),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        self.grafico_analise_html.load_html(pio.to_html(fig, full_html=False))

        # Atualiza os comboboxes de previsão
        times = self.df_processed['Time'].tolist()
        self.combo_time1['values'] = times
        self.combo_time2['values'] = times

    def _fazer_previsao(self):
        """Realiza a previsão e exibe resultados detalhados com estilização."""
        time1 = self.combo_time1.get()
        time2 = self.combo_time2.get()
        if not time1 or not time2 or time1 == time2:
            messagebox.showwarning("Aviso", "Selecione dois times diferentes!")
            return

        try:
            resultado = self.prediction_model.predict_match(time1, time2)
            self.resultado_text.delete(1.0, tk.END)

            # Cabeçalho principal
            self.resultado_text.insert(tk.END, f"Análise de Confronto: {time1} vs {time2}\n\n", "header")

            # Estatísticas do Time 1
            self.resultado_text.insert(tk.END, f"Estatísticas de {time1}\n", "subheader")
            self.resultado_text.insert(tk.END, f"  Aproveitamento: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time1']['aproveitamento']:.3f}%\n", "time1")
            self.resultado_text.insert(tk.END, f"  Saldo de Gols: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time1']['saldo_gols']:.0f}\n", "time1")
            self.resultado_text.insert(tk.END, f"  Média de Gols Pró: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time1']['media_gols_pro']:.3f}\n", "time1")
            self.resultado_text.insert(tk.END, f"  Média de Gols Contra: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time1']['media_gols_contra']:.3f}\n\n", "time1")

            # Estatísticas do Time 2
            self.resultado_text.insert(tk.END, f"Estatísticas de {time2}\n", "subheader")
            self.resultado_text.insert(tk.END, f"  Aproveitamento: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time2']['aproveitamento']:.3f}%\n", "time2")
            self.resultado_text.insert(tk.END, f"  Saldo de Gols: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time2']['saldo_gols']:.0f}\n", "time2")
            self.resultado_text.insert(tk.END, f"  Média de Gols Pró: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time2']['media_gols_pro']:.3f}\n", "time2")
            self.resultado_text.insert(tk.END, f"  Média de Gols Contra: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time2']['media_gols_contra']:.3f}\n\n", "time2")

            # Probabilidades
            self.resultado_text.insert(tk.END, "Probabilidades de Vitória\n", "subheader")
            self.resultado_text.insert(tk.END, f"  {time1}: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time1']['probabilidade']:.3f}% (±{resultado['time1']['intervalo_confianca']:.3f}%)\n", "time1")
            self.resultado_text.insert(tk.END, f"  {time2}: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time2']['probabilidade']:.3f}% (±{resultado['time2']['intervalo_confianca']:.3f}%)\n\n", "time2")

            # Odds Implícitas
            self.resultado_text.insert(tk.END, "Odds Implícitas\n", "subheader")
            self.resultado_text.insert(tk.END, f"  {time1}: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time1']['odds']:.2f}\n", "time1")
            self.resultado_text.insert(tk.END, f"  {time2}: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['time2']['odds']:.2f}\n\n", "time2")

            # Mercados Adicionais
            self.resultado_text.insert(tk.END, "Mercados Adicionais\n", "subheader")
            self.resultado_text.insert(tk.END, f"  Mais de 2.5 Gols: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['mercados']['mais_2_5_gols']:.3f}%\n", "normal")
            self.resultado_text.insert(tk.END, f"  Menos de 2.5 Gols: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['mercados']['menos_2_5_gols']:.3f}%\n", "normal")
            self.resultado_text.insert(tk.END, f"  Ambos Marcam: ", "normal")
            self.resultado_text.insert(tk.END, f"{resultado['mercados']['ambos_marcam']:.3f}%\n\n", "normal")

            # Sugestão de Aposta
            self.resultado_text.insert(tk.END, "Sugestão de Aposta\n", "subheader")
            self.resultado_text.insert(tk.END, f"  {resultado['sugestao']}\n", "normal")

            # Gráfico interativo de probabilidades
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=[time1, time2],
                x=[resultado['time1']['probabilidade'], resultado['time2']['probabilidade']],
                orientation='h',
                marker_color=['#00cc66', '#ff4444'],
                text=[f"{resultado['time1']['probabilidade']:.3f}%", f"{resultado['time2']['probabilidade']:.3f}%"],
                textposition='auto'
            ))
            fig.update_layout(
                title="Probabilidades de Vitória",
                title_font=dict(size=20, color='white'),
                paper_bgcolor='#121212',
                plot_bgcolor='#1e1e1e',
                font=dict(color='white'),
                xaxis_title="Probabilidade (%)",
                yaxis_title="Times",
                margin=dict(l=50, r=50, t=50, b=50)
            )
            self.grafico_previsao_html.load_html(pio.to_html(fig, full_html=False))
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao realizar previsão: {str(e)}")

    def run(self):
        """Inicia a interface gráfica."""
        self.root.mainloop()

def main():
    """Função principal."""
    logging.info("Iniciando aplicação...")
    api_key = "live_8a4d7e4aa5b0a67144xxxxxxxxxx1"  #your API here
    app = FootballAnalysisGUI(api_key=api_key)
    app.run()

if __name__ == "__main__":
    main()
