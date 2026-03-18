import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import quad
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Configuración ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PayFast Colombia — Análisis de Latencia",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero {
    background: linear-gradient(135deg, #021B2E 0%, #065A82 60%, #1C7293 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: rgba(2,195,154,0.15);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 30%;
    width: 150px; height: 150px;
    background: rgba(249,97,103,0.1);
    border-radius: 50%;
}
.hero-content {
    flex: 1;
    z-index: 2;
}
.hero-logo {
    flex: 0 0 auto;
    margin-left: 2rem;
    z-index: 2;
}
.hero h1 { color: #fff; font-size: 2rem; font-weight: 800; margin: 0; }
.hero p  { color: #AEDFF7; font-size: 1rem; margin: 0.4rem 0 0 0; }
.hero .badge {
    display: inline-block;
    background: rgba(2,195,154,0.2);
    border: 1px solid #02C39A;
    color: #02C39A;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.8rem;
}
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.07);
    border-top: 4px solid #065A82;
    text-align: center;
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card.red    { border-top-color: #F96167; }
.kpi-card.green  { border-top-color: #02C39A; }
.kpi-card.gold   { border-top-color: #F5A623; }
.kpi-card.purple { border-top-color: #8B5CF6; }
.kpi-label { font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 2rem; font-weight: 800; color: #021B2E; line-height: 1.1; margin: 0.3rem 0; }
.kpi-sub   { font-size: 0.78rem; color: #94A3B8; }

.step-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
    border-left: 5px solid #065A82;
}
.step-card.active { border-left-color: #F5A623; background: #FFFBF0; }

.formula-block {
    background: linear-gradient(135deg, #EAF4FB, #F0FDF9);
    border: 1.5px solid #AEDFF7;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    color: #065A82;
    font-weight: 500;
    text-align: center;
    margin: 0.6rem 0;
    box-shadow: inset 0 1px 4px rgba(6,90,130,0.08);
}
.explanation {
    background: #F8FAFC;
    border-left: 3px solid #02C39A;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #334155;
    margin: 0.4rem 0;
    line-height: 1.6;
}
.alert-success {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border: 1px solid #86EFAC;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #166534;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}
.alert-danger {
    background: linear-gradient(135deg, #FFF1F2, #FFE4E6);
    border: 1px solid #FCA5A5;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #991B1B;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}
.alert-info {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 1px solid #93C5FD;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    color: #1E40AF;
    font-size: 0.88rem;
    margin: 0.5rem 0;
}
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #021B2E;
    padding: 0.5rem 0;
    margin: 1rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.tab-intro {
    background: linear-gradient(135deg, #F8FAFC, #EFF6FF);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    font-size: 0.9rem;
    color: #334155;
    margin-bottom: 1rem;
    border-left: 4px solid #065A82;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #021B2E 0%, #065A82 100%);
}
div[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="stSidebar"] .stSlider > div > div { background: rgba(255,255,255,0.2) !important; }
div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
@media (max-width: 768px) {
    .info-grid {
        grid-template-columns: 1fr;
    }
    .hero {
        flex-direction: column;
    }
    .hero-logo {
        margin-left: 0;
        margin-top: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def f_exp(x, lam):
    return lam * np.exp(-lam * x)

def prob_mayor(umbral, lam):
    return float(np.exp(-lam * umbral))

def prob_intervalo(a, b, lam):
    result, _ = quad(f_exp, a, b, args=(lam,))
    return result

def estimar_lambda(d):
    return 1.0 / np.mean(d)

COLORES = {
    'navy':     '#065A82',
    'teal':     '#1C7293',
    'mint':     '#02C39A',
    'coral':    '#F96167',
    'gold':     '#F5A623',
    'purple':   '#8B5CF6',
    'dark':     '#021B2E',
    'gray':     '#64748B',
    'offwhite': '#F8FAFC'
}

# ─── Hero con Logo ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <h1>⚡ PayFast Colombia S.A.S.</h1>
        <p>Modelo probabilístico de latencia API · Cálculo Integral aplicado a sistemas digitales</p>
        <span class="badge">🎓 Fundación Universitaria Compensar · Cálculo Integral · 2026</span>
    </div>
    
</div>
""", unsafe_allow_html=True)

# ─── Carga de datos ────────────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader("📂 CSV (columna: response_time_s)", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if 'response_time_s' not in df.columns:
            st.error("El archivo debe tener columna 'response_time_s'")
            st.stop()
        datos = df['response_time_s'].dropna().values
        datos = datos[datos > 0]
        fuente = "📁 Datos cargados desde archivo"
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    # Datos simulados: 500 muestras Exp(λ=1) + 15 outliers uniformes [4,8]
    np.random.seed(42)
    datos = np.concatenate([
        np.random.exponential(1.0, 500),
        np.random.uniform(4, 8, 15)
    ])
    np.random.shuffle(datos)
    df = pd.DataFrame({
        'response_time_s': np.round(datos, 4),
        'timestamp': pd.date_range('2026-01-01 08:00', periods=len(datos), freq='1min')
    })
    fuente = "🔵 Datos simulados — PayFast Colombia (500 muestras Exp(λ=1) + 15 outliers)"

# Estimación MLE: λ̂ = 1/x̄
lam_est = estimar_lambda(datos)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel de control")
    st.markdown("---")

    st.markdown("### 📏 Umbral crítico")
    umbral = st.slider("Segundos", 1.0, 10.0, 3.0, 0.5,
                       help="Límite máximo aceptable de tiempo de respuesta (por defecto: 3s)")

    st.markdown("---")
    st.markdown("### 🔬 Escenario optimizado")
    st.markdown(
        f"<small>λ estimado (MLE): <b>{lam_est:.3f}</b>  →  E[X] = {1/lam_est:.3f}s</small>",
        unsafe_allow_html=True
    )
    lambda_opt = st.slider(
        "λ optimizado", 0.5, 6.0, 2.0, 0.1,
        help="Simula una mejora de infraestructura. Mayor λ = menor tiempo promedio."
    )
    st.markdown(
        f"<small>→ E[X] opt: <b>{1/lambda_opt:.3f}s</b> | "
        f"P(X>{umbral:g}s): <b>{prob_mayor(umbral, lambda_opt)*100:.3f}%</b></small>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("⚡ PayFast Colombia · Cálculo Integral · 2026")

# ─── Métricas derivadas ────────────────────────────────────────────────────────
p_actual  = prob_mayor(umbral, lam_est)
p_opt     = prob_mayor(umbral, lambda_opt)
reduccion = (p_actual - p_opt) / p_actual * 100 if p_actual > 0 else 0
E_actual  = 1.0 / lam_est
E_opt     = 1.0 / lambda_opt

st.info(fuente)

# ─── KPIs principales ─────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (c1, "default", "λ estimado (MLE)",         f"{lam_est:.3f}",      f"x̄ = {np.mean(datos):.3f}s"),
    (c2, "red",     f"P(X>{umbral:.0f}s) actual",   f"{p_actual*100:.2f}%", f"e⁻{umbral*lam_est:.2f}"),
    (c3, "green",   f"P(X>{umbral:.0f}s) optimizado",f"{p_opt*100:.3f}%",   f"λ = {lambda_opt:.1f}"),
    (c4, "gold",    "Reducción relativa",        f"{reduccion:.1f}%",   "impacto optimización"),
    (c5, "purple",  "E[X] actual",               f"{E_actual:.3f}s",    "tiempo promedio"),
    (c6, "default", "Registros analizados",      f"{len(datos):,}",     "transacciones"),
]
for col, cls, label, val, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs (3 pestañas sin Monte Carlo) ─────────────────────────────────────────
tabs = st.tabs([
    "📊 Análisis de Datos",
    "📈 Densidad y Probabilidad",
    "📚 Desarrollo Matemático",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — ANÁLISIS DE DATOS
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div class="tab-intro">
    📊 <b>¿Por qué empezamos aquí?</b><br>
    Antes de aplicar cualquier modelo matemático es necesario explorar los datos.
    Esta pestaña responde a la pregunta: <i>¿los datos observados son consistentes con una
    distribución exponencial?</i><br><br>
    El <b>histograma</b> muestra la distribución real de los tiempos de respuesta registrados.
    La <b>curva roja</b> es la distribución exponencial ajustada con el parámetro λ estimado
    mediante <b>Máxima Verosimilitud (MLE)</b>: λ̂ = 1/x̄.
    Si la curva se superpone bien al histograma, el modelo exponencial es válido.
    </div>""", unsafe_allow_html=True)

    col_hist, col_stats = st.columns([3, 2])

    with col_hist:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=datos, nbinsx=45, histnorm='probability density',
            marker=dict(
                color='rgba(6,90,130,0.45)',
                line=dict(color=COLORES['navy'], width=0.4)
            ),
            name='Datos observados'
        ))
        x_fit = np.linspace(0, min(np.max(datos) * 0.9, 14), 300)
        fig_h.add_trace(go.Scatter(
            x=x_fit, y=f_exp(x_fit, lam_est),
            mode='lines', line=dict(color=COLORES['coral'], width=2.5),
            name=f'Exp ajustada — λ̂ = {lam_est:.3f}'
        ))
        fig_h.add_vline(x=umbral, line=dict(color='#333', dash='dash', width=2))
        fig_h.add_annotation(
            x=umbral + 0.1, y=lam_est * 0.6,
            text=f"umbral = {umbral:.0f}s",
            showarrow=False, font=dict(size=10, color='#333'),
            bgcolor='rgba(255,255,255,0.85)'
        )
        fig_h.update_layout(
            title="Histograma de tiempos de respuesta vs. distribución exponencial ajustada",
            xaxis_title="Tiempo de respuesta (s)",
            yaxis_title="Densidad de probabilidad",
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=400, margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_stats:
        st.markdown('<div class="section-header">📐 Estadísticas descriptivas</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="explanation">
        El estimador MLE para la distribución exponencial es <b>λ̂ = 1/x̄</b>.
        Con x̄ = {np.mean(datos):.4f}s se obtiene λ̂ = {lam_est:.4f},
        lo que significa un tiempo promedio teórico de <b>E[X] = 1/λ = {E_actual:.3f}s</b>.
        </div>""", unsafe_allow_html=True)

        percentiles = [50, 75, 90, 95, 99]
        stats_df = pd.DataFrame({
            'Estadístico': [
                'N registros', 'Media (x̄)', 'Mediana', 'Desv. estándar',
                'Mínimo', 'Máximo',
                *[f'Percentil {p}' for p in percentiles],
                f'% empírico > {umbral:.0f}s',
                'λ̂ estimado (MLE)'
            ],
            'Valor': [
                f'{len(datos):,}',
                f'{np.mean(datos):.4f} s',
                f'{np.median(datos):.4f} s',
                f'{np.std(datos):.4f} s',
                f'{np.min(datos):.4f} s',
                f'{np.max(datos):.4f} s',
                *[f'{np.percentile(datos, p):.4f} s' for p in percentiles],
                f'{np.mean(datos > umbral)*100:.2f}%',
                f'{lam_est:.4f}'
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True, height=390)

    # Tabla de distribución por zonas
    st.markdown('<div class="section-header">📋 Distribución por zonas — comparación escenarios</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="explanation">
    Esta tabla replica la tabla del documento (escenario inicial λ=1 vs optimizado λ=2).
    Cada porcentaje es el área bajo la curva f(x) en ese intervalo, calculada con la integral definida.
    Las tres filas suman 100%.
    </div>""", unsafe_allow_html=True)

    zonas = ['P(X ≤ 1s)', 'P(1s < X ≤ 3s)', 'P(X > 3s)']
    col_tabla1, col_tabla2 = st.columns(2)
    with col_tabla1:
        st.markdown(f"**Escenario actual — λ = {lam_est:.3f}**")
        t1 = pd.DataFrame({
            'Intervalo': zonas,
            'Probabilidad': [
                f"{prob_intervalo(0, 1, lam_est)*100:.2f}%",
                f"{prob_intervalo(1, 3, lam_est)*100:.2f}%",
                f"{p_actual*100:.2f}%"
            ]
        })
        st.dataframe(t1, hide_index=True, use_container_width=True)
    with col_tabla2:
        st.markdown(f"**Escenario optimizado — λ = {lambda_opt:.1f}**")
        p_opt_z3 = prob_mayor(3.0, lambda_opt)
        t2 = pd.DataFrame({
            'Intervalo': zonas,
            'Probabilidad': [
                f"{prob_intervalo(0, 1, lambda_opt)*100:.2f}%",
                f"{prob_intervalo(1, 3, lambda_opt)*100:.2f}%",
                f"{p_opt_z3*100:.2f}%"
            ]
        })
        st.dataframe(t2, hide_index=True, use_container_width=True)

    st.markdown(f"""
    <div class="alert-info">
    📌 <b>Lectura:</b> La zona crítica P(X > 3s) pasa de <b>{p_actual*100:.2f}%</b> (actual)
    a <b>{p_opt*100:.3f}%</b> (optimizado). Eso es una reducción del <b>{reduccion:.1f}%</b> en riesgo relativo.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — DENSIDAD Y PROBABILIDAD (REDISEÑADA)
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("""
    <div class="tab-intro">
    📈 <b>El área bajo la curva es una probabilidad.</b><br>
    La función de densidad f(x) = λe<sup>−λx</sup> describe cómo se distribuyen los tiempos de respuesta.
    La probabilidad de que el tiempo supere el umbral es el <b>área bajo la curva a la derecha de x = umbral</b>,
    lo cual es exactamente lo que calcula la integral definida:<br><br>
    P(X > umbral) = ∫<sub>umbral</sub><sup>∞</sup> λe<sup>−λx</sup> dx = e<sup>−λ·umbral</sup><br><br>
    Mueve el umbral en el panel izquierdo para ver cómo cambia el área roja en tiempo real.
    La curva verde punteada muestra el sistema optimizado (λ mayor → curva más alta y estrecha → menos área crítica).
    </div>""", unsafe_allow_html=True)

    # Gráfica a ancho completo
    x_max = max(umbral * 3, 10)
    x = np.linspace(0, x_max, 600)

    xs = x[x <= umbral]
    xc = x[x >= umbral]

    fig = go.Figure()
    # Zona segura (verde)
    fig.add_trace(go.Scatter(
        x=np.concatenate([[xs[0]], xs, [xs[-1]]]),
        y=np.concatenate([[0], f_exp(xs, lam_est), [0]]),
        fill='toself', fillcolor='rgba(2,195,154,0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'Zona segura — P(X ≤ {umbral:.0f}s) = {(1-p_actual)*100:.2f}%'
    ))
    # Zona crítica (rojo)
    fig.add_trace(go.Scatter(
        x=np.concatenate([[xc[0]], xc, [xc[-1]]]),
        y=np.concatenate([[0], f_exp(xc, lam_est), [0]]),
        fill='toself', fillcolor='rgba(249,97,103,0.45)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'⚠️ Zona crítica — P(X > {umbral:.0f}s) = {p_actual*100:.3f}%'
    ))
    # Curva actual (azul)
    fig.add_trace(go.Scatter(
        x=x, y=f_exp(x, lam_est), mode='lines',
        line=dict(color=COLORES['navy'], width=3),
        name=f'f(x) actual — λ = {lam_est:.3f}'
    ))
    # Curva optimizada (verde punteada)
    fig.add_trace(go.Scatter(
        x=x, y=f_exp(x, lambda_opt), mode='lines',
        line=dict(color=COLORES['mint'], width=2, dash='dot'),
        name=f'f(x) optimizada — λ = {lambda_opt:.1f}'
    ))
    fig.add_vline(x=umbral, line=dict(color='#333', dash='dash', width=2))
    fig.add_annotation(
        x=umbral + 0.1, y=lam_est * 0.75,
        text=f"<b>x = {umbral:.0f}s</b><br>umbral crítico",
        showarrow=False, font=dict(size=11, color='#333'),
        bgcolor='rgba(255,255,255,0.8)', bordercolor='#ccc', borderwidth=1
    )
    # Anotación del área roja
    x_mid = umbral + (x_max - umbral) / 3
    fig.add_annotation(
        x=x_mid, y=f_exp(x_mid, lam_est) * 0.5 + 0.05,
        text=f"<b>P(X > {umbral:.0f}s) = {p_actual*100:.3f}%</b><br>= e^(−{umbral:.0f}×{lam_est:.3f})",
        showarrow=True, arrowhead=2, arrowcolor=COLORES['coral'],
        font=dict(size=11, color=COLORES['coral']),
        bgcolor='rgba(255,255,255,0.9)', bordercolor=COLORES['coral'], borderwidth=1
    )
    fig.update_layout(
        title=dict(
            text=f"Función de Densidad Exponencial  |  λ = {lam_est:.3f}  |  E[X] = {E_actual:.3f}s",
            font=dict(size=14, color=COLORES['dark'])
        ),
        xaxis_title="Tiempo de respuesta x (segundos)",
        yaxis_title="Densidad de probabilidad f(x)",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0.5, xanchor='center'),
        plot_bgcolor='#FAFAFA', paper_bgcolor='white',
        height=500, margin=dict(l=60, r=40, t=80, b=60)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Información debajo de la gráfica en dos columnas
    st.markdown('<div class="section-header">📊 Análisis de Probabilidad Crítica</div>', unsafe_allow_html=True)
    
    col_formula, col_riesgo = st.columns(2)
    
    with col_formula:
        st.markdown('<div class="section-header">🔢 Solución Analítica</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation">
        Esta es la integral resuelta analíticamente. El área roja de la gráfica
        corresponde exactamente a este número.
        </div>""", unsafe_allow_html=True)
        st.markdown(
            f'<div class="formula-block">'
            f'P(X > {umbral:.0f}s)<br>'
            f'= ∫_{umbral:.0f}^∞ λe^(−λx) dx<br>'
            f'= e^(−{umbral:.0f}×{lam_est:.3f})<br>'
            f'= e^(−{umbral*lam_est:.3f})<br>'
            f'= <b>{p_actual:.6f}</b>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_riesgo:
        st.markdown('<div class="section-header">⚠️ Clasificación de Riesgo</div>', unsafe_allow_html=True)
        if p_actual > 0.05:
            st.markdown(
                f'<div class="alert-danger">⚠️ <b>Riesgo elevado</b><br>'
                f'{p_actual*100:.2f}% de las transacciones superan el umbral.</div>',
                unsafe_allow_html=True
            )
        elif p_actual > 0.01:
            st.markdown(
                f'<div class="alert-info">📋 <b>Riesgo moderado</b><br>'
                f'{p_actual*100:.2f}% de transacciones afectadas.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="alert-success">✅ <b>Riesgo bajo</b><br>'
                f'{p_actual*100:.3f}% — sistema dentro de parámetros aceptables.</div>',
                unsafe_allow_html=True
            )

    # Verificación y comparación
    st.markdown('<div class="section-header">🔍 Verificación Numérica con scipy.integrate.quad</div>', unsafe_allow_html=True)
    col_verify, col_compare = st.columns(2)
    
    with col_verify:
        st.markdown("""
        <div class="explanation">
        Validamos el resultado analítico con <b>scipy.integrate.quad</b> (cuadratura de Gauss).
        Si coinciden, el modelo es sólido.
        </div>""", unsafe_allow_html=True)
        p_scipy = prob_intervalo(umbral, 1000, lam_est)
        st.markdown(
            f'<div class="alert-success">'
            f'<b>scipy.quad</b> = {p_scipy:.6f}<br>'
            f'<b>Analítico</b> = {p_actual:.6f}<br>'
            f'<b>Diferencia</b> = {abs(p_scipy - p_actual):.2e} ✓'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_compare:
        st.markdown('<div class="section-header">📉 Comparación Escenarios</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            'Escenario': [f'Actual (λ={lam_est:.3f})', f'Optimizado (λ={lambda_opt:.1f})'],
            f'P(X>{umbral:.0f}s)': [f'{p_actual*100:.4f}%', f'{p_opt*100:.4f}%'],
            'E[X]': [f'{E_actual:.3f}s', f'{E_opt:.3f}s']
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.markdown(
            f'<div class="alert-success">🚀 <b>Reducción de riesgo: {reduccion:.1f}%</b></div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — DESARROLLO MATEMÁTICO PASO A PASO
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""
    <div class="tab-intro">
    📚 <b>Desarrollo matemático completo — 4 pasos.</b><br>
    Esta pestaña muestra el camino riguroso desde la verificación del modelo hasta el cálculo
    del valor esperado. Los pasos siguen exactamente la metodología planteada en el proyecto:<br><br>
    <b>Paso 1</b>: Verificar que f(x) es una FDA válida (integral impropia = 1)<br>
    <b>Paso 2</b>: Plantear y resolver P(X > umbral) analíticamente<br>
    <b>Paso 3</b>: Comparar los dos escenarios (λ actual vs λ optimizado)<br>
    <b>Paso 4</b>: Calcular E[X] mediante integración por partes<br><br>
    Usa el modo <i>Paso a paso</i> durante la sustentación para avanzar de uno en uno.
    </div>""", unsafe_allow_html=True)

    modo = st.radio(
        "Modo de visualización:",
        ["📖 Paso a paso", "📄 Desarrollo completo"],
        horizontal=True
    )

    # Alias para las fórmulas
    u       = umbral
    u_fmt   = f"{u:g}"
    lf      = lam_est
    lf3     = f"{lf:.3f}"
    prod    = u * lf
    e_val   = p_actual

    pasos = [
        {
            "titulo": "Paso 1 — Verificación de la función de densidad",
            "concepto": (
                f"Antes de calcular probabilidades, debemos verificar que f(x) = {lf3}·e^(−{lf3}·x) "
                f"es una función de densidad válida. Debe cumplir dos condiciones: "
                f"f(x) ≥ 0 para todo x ≥ 0, y que su integral en todo el dominio sea igual a 1."
            ),
            "formula": f"∫₀^∞ {lf3}·e^(−{lf3}·x) dx = ?",
            "desarrollo": [
                ("Aplicamos la definición de integral impropia como límite:",
                 f"= lim[b→∞] ∫₀ᵇ {lf3}·e^(−{lf3}x) dx"),
                ("La antiderivada de λe^(−λx) es −e^(−λx):",
                 f"= lim[b→∞] [ −e^(−{lf3}x) ]₀ᵇ"),
                ("Evaluamos en los límites b y 0:",
                 f"= lim[b→∞] (−e^(−{lf3}·b)) − (−e⁰)"),
                (f"Cuando b→∞, e^(−{lf3}·b) → 0. Y e⁰ = 1:",
                 "= 0 + 1 = 1  ✓"),
            ],
            "conclusion": (
                f"f(x) = {lf3}·e^(−{lf3}·x) es una FDA válida: "
                f"f(x) ≥ 0 para todo x ≥ 0 y la integral total es exactamente 1."
            ),
            "color": COLORES['navy']
        },
        {
            "titulo": f"Paso 2 — Resolución analítica de P(X > {u_fmt}s)",
            "concepto": (
                f"La probabilidad de que el tiempo de respuesta supere {u_fmt}s es el área bajo la curva "
                f"a la derecha de x = {u_fmt}. Resolvemos esta integral impropia aplicando el "
                f"Teorema Fundamental del Cálculo."
            ),
            "formula": f"P(X > {u_fmt}) = ∫_{u_fmt}^∞ {lf3}·e^(−{lf3}x) dx",
            "desarrollo": [
                ("Escribimos como integral impropia con límite superior → ∞:",
                 f"= lim[b→∞] ∫_{u_fmt}ᵇ {lf3}·e^(−{lf3}x) dx"),
                ("Evaluamos la antiderivada −e^(−λx) en los límites:",
                 f"= lim[b→∞] [ −e^(−{lf3}x) ]_{u_fmt}^b"),
                ("Límite superior: cuando x→∞, e^(−λx) → 0:",
                 "lim[b→∞] (−e^(−λb)) = 0"),
                (f"Límite inferior: evaluamos en x = {u_fmt}:",
                 f"−(−e^(−{lf3}×{u_fmt})) = e^(−{prod:.4f})"),
                ("Resultado final:",
                 f"P(X > {u_fmt}) = e^(−{lf3}×{u_fmt}) = e^(−{prod:.4f}) = {e_val:.6f}"),
            ],
            "conclusion": (
                f"P(X > {u_fmt}s) = e^(−{prod:.4f}) = {e_val:.6f}. "
                f"El {e_val*100:.3f}% de las transacciones superan el umbral de {u_fmt}s "
                f"con el sistema actual (λ = {lf3})."
            ),
            "color": COLORES['gold']
        },
        {
            "titulo": f"Paso 3 — Comparación de escenarios (umbral = {u_fmt}s)",
            "concepto": (
                f"El documento compara dos escenarios: el estado inicial (λ estimado de los datos) "
                f"y un estado optimizado (mayor λ). Aumentar λ reduce directamente P(X > {u_fmt}s) "
                f"porque la función e^(−λ·umbral) es decreciente en λ."
            ),
            "formula": (
                f"Actual:     P(X>{u_fmt}) = e^(−{lf3}×{u_fmt}) = {p_actual:.6f}<br>"
                f"Optimizado: P(X>{u_fmt}) = e^(−{lambda_opt:.1f}×{u_fmt}) = {p_opt:.6f}"
            ),
            "desarrollo": [
                (f"Estado actual (λ = {lf3}, E[X] = {E_actual:.3f}s):",
                 f"P(X>{u_fmt}) = {p_actual*100:.4f}%  →  {p_actual*10000:.0f} transacciones críticas por cada 10,000"),
                (f"Estado optimizado (λ = {lambda_opt:.1f}, E[X] = {E_opt:.3f}s):",
                 f"P(X>{u_fmt}) = {p_opt*100:.4f}%  →  {p_opt*10000:.0f} transacciones críticas por cada 10,000"),
                ("Reducción absoluta:",
                 f"Δ = {p_actual*100:.4f}% − {p_opt*100:.4f}% = {(p_actual-p_opt)*100:.4f} pp"),
                ("Reducción relativa:",
                 f"({p_actual:.6f} − {p_opt:.6f}) / {p_actual:.6f} × 100 = {reduccion:.2f}%"),
            ],
            "conclusion": (
                f"Pasar de λ = {lf3} a λ = {lambda_opt:.1f} reduce el riesgo en {reduccion:.1f}%, "
                f"equivalente a rescatar {(p_actual-p_opt)*10000:.0f} transacciones por cada 10,000."
            ),
            "color": COLORES['coral']
        },
        {
            "titulo": "Paso 4 — Valor esperado por integración por partes",
            "concepto": (
                "El valor esperado E[X] representa el tiempo promedio de respuesta del sistema. "
                "La integral ∫₀^∞ x·f(x)dx no puede resolverse directamente — "
                "requiere integración por partes: ∫u·dv = uv − ∫v·du."
            ),
            "formula": f"E[X] = ∫₀^∞ x · {lf3}·e^(−{lf3}x) dx",
            "desarrollo": [
                ("Identificamos u y dv:",
                 f"u = x  →  du = dx<br>dv = {lf3}·e^(−{lf3}x)dx  →  v = −e^(−{lf3}x)"),
                ("Aplicamos ∫u·dv = uv − ∫v·du:",
                 f"= [−x·e^(−{lf3}x)]₀^∞  +  ∫₀^∞ e^(−{lf3}x) dx"),
                (f"Primer término: lim[x→∞] x·e^(−{lf3}x) = 0 (la exponencial domina):",
                 "= 0"),
                ("Segundo término, integral de e^(−λx):",
                 f"∫₀^∞ e^(−{lf3}x) dx = 1/{lf3} = {E_actual:.4f}"),
                ("Resultado:",
                 f"E[X] = 1/λ = 1/{lf3} = {E_actual:.4f} segundos"),
            ],
            "conclusion": (
                f"E[X] = 1/λ = 1/{lf3} = {E_actual:.4f}s — tiempo promedio actual. "
                f"Tras la optimización (λ = {lambda_opt:.1f}): E[X] = {E_opt:.4f}s."
            ),
            "color": COLORES['mint']
        },
    ]

    if modo == "📖 Paso a paso":
        paso_actual = st.slider("Selecciona el paso:", 1, len(pasos), 1) - 1
        paso = pasos[paso_actual]

        col_prev, col_ind, col_next = st.columns([1, 3, 1])
        with col_prev:
            if paso_actual > 0 and st.button("← Anterior"):
                st.session_state['paso'] = paso_actual - 1
        with col_ind:
            st.markdown(
                f"<p style='text-align:center;color:{COLORES['gray']};font-weight:600;'>"
                f"Paso {paso_actual+1} de {len(pasos)}</p>",
                unsafe_allow_html=True
            )
        with col_next:
            if paso_actual < len(pasos) - 1 and st.button("Siguiente →"):
                st.session_state['paso'] = paso_actual + 1

        st.markdown(f"""
        <div class="step-card active">
            <h3 style="color:{paso['color']};margin:0 0 0.5rem 0;">{paso['titulo']}</h3>
            <div class="explanation">{paso['concepto']}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔢 Planteamiento</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="formula-block">{paso["formula"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">📝 Desarrollo</div>', unsafe_allow_html=True)
        for i, (desc, form) in enumerate(paso['desarrollo']):
            cd, cf = st.columns([1, 1])
            with cd:
                st.markdown(
                    f'<div class="explanation"><b>{i+1}.</b> {desc}</div>',
                    unsafe_allow_html=True
                )
            with cf:
                st.markdown(
                    f'<div class="formula-block" style="font-size:0.95rem;">{form}</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            f'<div class="alert-success">💡 <b>Conclusión:</b> {paso["conclusion"]}</div>',
            unsafe_allow_html=True
        )

    else:
        for i, paso in enumerate(pasos):
            icon = "✅" if i < 3 else "🔢"
            with st.expander(f"{icon} {paso['titulo']}", expanded=(i == 0)):
                st.markdown(
                    f'<div class="explanation">{paso["concepto"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="formula-block">{paso["formula"]}</div>',
                    unsafe_allow_html=True
                )
                for desc, form in paso['desarrollo']:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            f'<div class="explanation">{desc}</div>',
                            unsafe_allow_html=True
                        )
                    with c2:
                        st.markdown(
                            f'<div class="formula-block" style="font-size:0.9rem;">{form}</div>',
                            unsafe_allow_html=True
                        )
                st.markdown(
                    f'<div class="alert-success">💡 {paso["conclusion"]}</div>',
                    unsafe_allow_html=True
                )


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94A3B8; font-size:0.82rem; padding:0.5rem;'>
⚡ PayFast Colombia S.A.S. · Análisis Probabilístico de Latencia API ·
Fundación Universitaria Compensar · Cálculo Integral · 2026<br>
<span style='color:#02C39A;'>
    Modelo: f(x) = λe^(−λx) · P(X>3) = e^(−3λ) · E[X] = 1/λ (integración por partes)
</span>
</div>
""", unsafe_allow_html=True)

# python -m streamlit run app_corregido.py