import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import expon, norm, gamma
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #021B2E 0%, #065A82 60%, #1C7293 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
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
.step-num {
    display: inline-block;
    background: #065A82;
    color: white;
    width: 28px; height: 28px;
    border-radius: 50%;
    text-align: center;
    line-height: 28px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 0.5rem;
}
.formula-block {
    background: linear-gradient(135deg, #EAF4FB, #F0FDF9);
    border: 1.5px solid #AEDFF7;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-family: 'Courier New', monospace;
    font-size: 1.05rem;
    color: #065A82;
    font-weight: 700;
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
.montecarlo-counter {
    font-size: 3rem;
    font-weight: 800;
    color: #065A82;
    text-align: center;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #021B2E 0%, #065A82 100%);
}
div[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="stSidebar"] .stSlider > div > div { background: rgba(255,255,255,0.2) !important; }
div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def f_exp(x, lam):    return lam * np.exp(-lam * x)
def f_norm(x, mu, s): return norm.pdf(x, mu, s)
def f_gamma(x, a, b): return gamma.pdf(x, a, scale=b)

def prob_mayor(umbral, lam):
    return float(np.exp(-lam * umbral))

def prob_intervalo(a, b, fn, *args):
    r, _ = quad(fn, a, b, args=args)
    return r

def estimar_lambda(d): return 1.0 / np.mean(d)

COLORES = {
    'navy': '#065A82', 'teal': '#1C7293', 'mint': '#02C39A',
    'coral': '#F96167', 'gold': '#F5A623', 'purple': '#8B5CF6',
    'dark': '#021B2E', 'gray': '#64748B', 'offwhite': '#F8FAFC'
}

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚡ PayFast Colombia S.A.S.</h1>
    <p>Modelo probabilístico de latencia API · Cálculo Integral aplicado a sistemas digitales</p>
    <span class="badge">🎓 Fundación Universitaria Compensar · Cálculo Integral · 2026</span>
</div>
""", unsafe_allow_html=True)

# ─── Paso 1: carga de datos ANTES del sidebar (necesario para lam_est) ─────────
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
        st.error(f"Error: {e}"); st.stop()
else:
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
    fuente = "🔵 Datos simulados — PayFast Colombia (escenario demostrativo)"

lam_est = estimar_lambda(datos)   # disponible para el sidebar

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Panel de control")
    st.markdown("---")

    st.markdown("### 📏 Umbral crítico")
    umbral = st.slider("Segundos", 1.0, 10.0, 3.0, 0.5)

    st.markdown("---")
    st.markdown("### 🔬 Escenario optimizado")
    st.markdown(
        f"<small>λ actual estimado: <b>{lam_est:.3f}</b> → E[X] = {1/lam_est:.3f}s</small>",
        unsafe_allow_html=True)
    lambda_opt = st.slider("λ optimizado", 0.5, 6.0, 2.0, 0.1,
                           help="Mayor λ = menor tiempo promedio = menor riesgo. Representa una mejora de infraestructura.")
    st.markdown(
        f"<small>→ E[X] opt: <b>{1/lambda_opt:.3f}s</b> | "
        f"P(X>{umbral:g}s): <b>{prob_mayor(umbral, lambda_opt)*100:.3f}%</b></small>",
        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💰 Impacto de negocio")
    transacciones_dia = st.number_input("Transacciones/día", 100, 1000000, 10000, 1000)
    valor_promedio    = st.number_input("Valor promedio ($COP)", 1000, 10000000, 150000, 5000)
    tasa_abandono     = st.slider("Tasa de abandono por latencia", 0.1, 1.0, 0.65, 0.05,
                                  help="% de usuarios que abandonan cuando hay latencia crítica")

    st.markdown("---")
    st.markdown("### 📐 Intervalo personalizado")
    a_custom = st.number_input("Límite inferior a", 0.0, 20.0, 1.0, 0.5)
    b_custom = st.number_input("Límite superior b", 0.0, 50.0, 3.0, 0.5)

    st.markdown("---")
    st.caption("⚡ PayFast Colombia · Cálculo Integral · 2026")

# ─── Métricas derivadas (después de sidebar para tener umbral y lambda_opt) ────
p_actual  = prob_mayor(umbral, lam_est)
p_opt     = prob_mayor(umbral, lambda_opt)
reduccion = (p_actual - p_opt) / p_actual * 100 if p_actual > 0 else 0
p_emp     = float(np.mean(datos > umbral))
E_actual  = 1.0 / lam_est
E_opt     = 1.0 / lambda_opt

st.info(fuente)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    (c1, "default", "λ estimado (MLE)", f"{lam_est:.3f}", f"x̄ = {np.mean(datos):.3f}s"),
    (c2, "red",     f"P(X>{umbral:.0f}s) actual",     f"{p_actual*100:.2f}%",   f"e⁻{umbral*lam_est:.2f}"),
    (c3, "green",   f"P(X>{umbral:.0f}s) optimizado", f"{p_opt*100:.3f}%",      f"λ={lambda_opt:.1f}"),
    (c4, "gold",    "Reducción relativa",  f"{reduccion:.1f}%",  "impacto optimización"),
    (c5, "purple",  "E[X] actual",         f"{E_actual:.3f}s",   "tiempo promedio"),
    (c6, "default", "Registros analizados",f"{len(datos):,}",    "transacciones"),
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

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📈 Densidad y Probabilidad",
    "📚 Desarrollo Matemático",
    "🎲 Simulador Monte Carlo",
    "⚖️ Comparación de Modelos",
    "💰 Impacto de Negocio",
    "📊 Análisis de Datos",
    "📐 Intervalo Personalizado",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DENSIDAD Y PROBABILIDAD
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div class="tab-intro">
    🎯 <b>¿Qué muestra esta pestaña?</b><br>
    Aquí visualizamos la <b>función de densidad de probabilidad (FDA)</b> del modelo exponencial ajustado a los datos de PayFast.
    Cada punto de la curva f(x) indica qué tan probable es que el tiempo de respuesta sea exactamente x segundos.
    <br><br>
    El <b>área bajo la curva</b> en cualquier intervalo es una probabilidad — esto es precisamente lo que calcula la integral definida.
    El área roja a la derecha del umbral representa <b>P(X > umbral)</b>: la fracción de transacciones con latencia crítica.
    Mueve el umbral en el panel izquierdo y observa cómo cambia el área en tiempo real.
    </div>""", unsafe_allow_html=True)

    col_graf, col_info = st.columns([3, 1])

    with col_graf:
        x_max = max(umbral * 3, 10)
        x = np.linspace(0, x_max, 600)
        y = f_exp(x, lam_est)

        xs = x[x <= umbral];  ys = f_exp(xs, lam_est)
        xc = x[x >= umbral];  yc = f_exp(xc, lam_est)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([[xs[0]], xs, [xs[-1]]]),
            y=np.concatenate([[0], ys, [0]]),
            fill='toself', fillcolor='rgba(2,195,154,0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'Zona segura — P(X≤{umbral:.0f}s) = {(1-p_actual)*100:.2f}%'
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([[xc[0]], xc, [xc[-1]]]),
            y=np.concatenate([[0], yc, [0]]),
            fill='toself', fillcolor='rgba(249,97,103,0.45)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'⚠️ Zona crítica — P(X>{umbral:.0f}s) = {p_actual*100:.3f}%'
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color=COLORES['navy'], width=3),
            name=f'f(x) = {lam_est:.3f}·e^(-{lam_est:.3f}·x)'
        ))
        # Curva optimizada
        fig.add_trace(go.Scatter(
            x=x, y=f_exp(x, lambda_opt), mode='lines',
            line=dict(color=COLORES['mint'], width=2, dash='dot'),
            name=f'f(x) optimizada — λ={lambda_opt:.1f}'
        ))
        fig.add_vline(x=umbral, line=dict(color='#333', dash='dash', width=2))
        fig.add_annotation(
            x=umbral + 0.1, y=lam_est * 0.75,
            text=f"<b>x = {umbral:.0f}s</b><br>umbral crítico",
            showarrow=False, font=dict(size=11, color='#333'),
            bgcolor='rgba(255,255,255,0.8)', bordercolor='#ccc', borderwidth=1
        )
        # Anotación del área
        x_mid = umbral + (x_max - umbral) / 3
        y_mid = f_exp(x_mid, lam_est) * 0.5
        fig.add_annotation(
            x=x_mid, y=y_mid + 0.05,
            text=f"<b>P(X>{umbral:.0f}) = {p_actual*100:.3f}%</b><br>= e^(-{umbral:.0f}×{lam_est:.3f})",
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
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            height=430, margin=dict(l=50, r=20, t=70, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">📊 Probabilidad crítica</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation">
        Esta es la integral que resolvimos analíticamente. El área roja de la gráfica
        corresponde exactamente a este número — la fracción de transacciones que tardan
        más del umbral aceptable.
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="formula-block">P(X > {umbral:.0f}) = e⁻{umbral:.0f}ˡ<br>= e⁻{umbral*lam_est:.3f}<br>= {p_actual:.6f}</div>', unsafe_allow_html=True)

        if p_actual > 0.05:
            st.markdown(f'<div class="alert-danger">⚠️ <b>Riesgo elevado</b><br>{p_actual*100:.2f}% de las transacciones superan el umbral. Se recomienda revisar la infraestructura.</div>', unsafe_allow_html=True)
        elif p_actual > 0.01:
            st.markdown(f'<div class="alert-info">📋 <b>Riesgo moderado</b><br>{p_actual*100:.2f}% de transacciones afectadas. Considerar optimización preventiva.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-success">✅ <b>Riesgo bajo</b><br>{p_actual*100:.3f}% — el sistema opera dentro de los parámetros aceptables.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">🔢 Distribución por zonas</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation">
        Cada fila es el área bajo la curva en ese intervalo, calculada con la integral
        definida. Las tres filas deben sumar 100%.
        </div>""", unsafe_allow_html=True)
        tabla_dist = pd.DataFrame({
            'Intervalo': [f'P(X ≤ 1s)', f'P(1 < X ≤ {umbral:.0f}s)', f'P(X > {umbral:.0f}s)'],
            '%': [
                f"{prob_intervalo(0, 1, f_exp, lam_est)*100:.1f}%",
                f"{prob_intervalo(1, umbral, f_exp, lam_est)*100:.1f}%",
                f"{p_actual*100:.2f}%"
            ]
        })
        st.dataframe(tabla_dist, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-header">🔍 Verificación doble</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation">
        Calculamos el mismo resultado de dos formas independientes para confirmar que
        el modelo es correcto: <b>analíticamente</b> (resolviendo la integral a mano)
        y <b>numéricamente</b> (usando el algoritmo de cuadratura de Gauss de SciPy).
        Si coinciden, el modelo es sólido.
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="alert-success">scipy.quad = <b>{prob_intervalo(umbral, 1000, f_exp, lam_est):.6f}</b><br>Analítico = <b>{p_actual:.6f}</b> ✓</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — DESARROLLO MATEMÁTICO PASO A PASO
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("""
    <div class="tab-intro">
    📚 <b>Desarrollo matemático completo.</b> Esta pestaña muestra el camino desde el planteamiento del problema
    hasta la fórmula final, paso a paso. Cada bloque tiene tres partes: el <b>concepto teórico</b> que lo justifica,
    la <b>expresión matemática</b> a resolver y el <b>desarrollo</b> que lleva al resultado.<br><br>
    Usa el modo <i>Paso a paso</i> para ir avanzando de a uno durante la sustentación,
    o <i>Desarrollo completo</i> para ver todo el proceso de una vez.
    </div>""", unsafe_allow_html=True)

    modo = st.radio("Modo de visualización:", ["📖 Paso a paso", "📄 Desarrollo completo"], horizontal=True)

    # Valores numéricos ya evaluados — se recalculan cada vez que cambia el umbral
    u      = umbral          # alias corto
    u_fmt  = f"{u:g}"       # muestra "3" o "2.5" sin ceros innecesarios
    lexp   = lam_est
    lexp3f = f"{lexp:.3f}"
    prod   = u * lexp        # λ × umbral
    e_val  = p_actual        # e^(-λu)

    pasos = [
        {
            "titulo": "Paso 1 — Verificación de la función de densidad",
            "concepto": f"Para que f(x) sea una función de densidad válida debe cumplir dos condiciones: ser no negativa en todo su dominio y que su integral en todo el dominio sea igual a 1. Verificamos esto con λ = {lexp3f}.",
            "formula": f"∫₀^∞ {lexp3f}·e^(-{lexp3f}·x) dx = ?",
            "desarrollo": [
                ("Aplicamos la definición de integral impropia como límite:",
                 f"∫₀^∞ λe⁻ˡˣ dx = lim[b→∞] ∫₀ᵇ {lexp3f}·e^(-{lexp3f}x) dx"),
                ("Calculamos la antiderivada de λe⁻ˡˣ:",
                 f"= lim[b→∞] [ -e⁻ˡˣ ]₀ᵇ"),
                ("Evaluamos el límite superior (b → ∞) y el inferior (x = 0):",
                 f"= lim[b→∞] (-e^(-{lexp3f}·b)) − (−e⁰)"),
                (f"Como e^(-{lexp3f}·b) → 0 cuando b → ∞:",
                 "= 0 + 1 = 1  ✓"),
            ],
            "conclusion": f"✅ f(x) = {lexp3f}·e^(-{lexp3f}·x) es una FDA válida: f(x) ≥ 0 para todo x ≥ 0 y la integral total es exactamente 1.",
            "color": COLORES['navy']
        },
        {
            "titulo": f"Paso 2 — Planteamiento: P(X > {u_fmt})",
            "concepto": f"La probabilidad de que el tiempo de respuesta supere el umbral de {u_fmt}s es el área bajo la curva de f(x) a la derecha de x = {u_fmt}. Como el límite superior es infinito, se trata de una integral impropia.",
            "formula": f"P(X > {u_fmt}) = ∫_{u_fmt}^∞ {lexp3f}·e^(-{lexp3f}·x) dx",
            "desarrollo": [
                ("La probabilidad en una FDA es el área bajo la curva en el intervalo:",
                 f"P(X > {u_fmt}) = ∫_{u_fmt}^∞ f(x) dx"),
                ("Sustituimos f(x) = λe⁻ˡˣ con λ = {:.3f}:".format(lexp),
                 f"= ∫_{u_fmt}^∞ {lexp3f}·e^(-{lexp3f}x) dx"),
                ("Escribimos como integral impropia (límite superior → ∞):",
                 f"= lim[b→∞] ∫_{u_fmt}ᵇ {lexp3f}·e^(-{lexp3f}x) dx"),
            ],
            "conclusion": f"El área roja en la gráfica de densidad (pestaña 1) es exactamente esta integral — la fracción de transacciones que tardan más de {u_fmt} segundos.",
            "color": COLORES['teal']
        },
        {
            "titulo": f"Paso 3 — Resolución analítica de P(X > {u_fmt})",
            "concepto": "Aplicamos el Teorema Fundamental del Cálculo: encontramos la antiderivada de la función y evaluamos en los límites. El límite en infinito se maneja con la definición de integral impropia.",
            "formula": f"∫ {lexp3f}·e^(-{lexp3f}x) dx = -e^(-{lexp3f}x) + C",
            "desarrollo": [
                ("Antiderivada de λe⁻ˡˣ (verificable derivando el resultado):",
                 f"-e^(-{lexp3f}x) + C"),
                (f"Evaluamos la antiderivada en los límites [{u_fmt}, ∞):",
                 f"[ -e^(-{lexp3f}x) ]_{u_fmt}^∞"),
                ("Límite superior: cuando x → ∞, e⁻ˡˣ → 0:",
                 "lim[x→∞] (−e^(-λx)) = 0"),
                (f"Límite inferior: evaluamos en x = {u_fmt}:",
                 f"−(−e^(-{lexp3f}×{u_fmt})) = e^(-{prod:.4f})"),
                (f"Resultado final:",
                 f"P(X > {u_fmt}) = e^(-{lexp3f}×{u_fmt}) = e^(-{prod:.4f}) = {e_val:.6f}"),
            ],
            "conclusion": f"🎯 P(X > {u_fmt}) = e^(-{prod:.4f}) = {e_val:.6f} → el {e_val*100:.3f}% de las transacciones superan el umbral de {u_fmt}s con el sistema actual (λ = {lexp3f}).",
            "color": COLORES['gold']
        },
        {
            "titulo": "Paso 4 — Valor esperado por integración por partes",
            "concepto": "El valor esperado E[X] mide el tiempo promedio de respuesta del sistema. La integral ∫₀^∞ x·f(x)dx no puede resolverse directamente — requiere integración por partes: ∫u·dv = uv − ∫v·du.",
            "formula": f"E[X] = ∫₀^∞ x · {lexp3f}·e^(-{lexp3f}x) dx",
            "desarrollo": [
                ("Identificamos u y dv para aplicar integración por partes:",
                 f"u = x  →  du = dx\ndv = {lexp3f}·e^(-{lexp3f}x)dx  →  v = −e^(-{lexp3f}x)"),
                ("Aplicamos la fórmula ∫u·dv = uv − ∫v·du:",
                 f"E[X] = [−x·e^(-{lexp3f}x)]₀^∞  +  ∫₀^∞ e^(-{lexp3f}x) dx"),
                (f"Primer término: lim[x→∞] x·e^(-{lexp3f}x) = 0 (la exponencial domina al polinomio):",
                 "= 0 − 0 = 0"),
                ("Segundo término: integral de e⁻ˡˣ:",
                 f"∫₀^∞ e^(-{lexp3f}x) dx = [−e^(-{lexp3f}x)/{lexp3f}]₀^∞ = 1/{lexp3f}"),
                ("Resultado:",
                 f"E[X] = 0 + 1/{lexp3f} = {E_actual:.4f} segundos"),
            ],
            "conclusion": f"✅ E[X] = 1/λ = 1/{lexp3f} = {E_actual:.4f}s — el tiempo promedio de respuesta del sistema actual. Con el sistema optimizado (λ={lambda_opt:.1f}): E[X] = {E_opt:.4f}s.",
            "color": COLORES['mint']
        },
        {
            "titulo": f"Paso 5 — Comparación de escenarios (umbral = {u_fmt}s)",
            "concepto": f"Aumentar λ (mejorar el sistema) reduce directamente P(X > {u_fmt}). Comparamos el estado actual con el escenario optimizado para cuantificar el impacto de la mejora.",
            "formula": (
                f"Actual:    P(X>{u_fmt}) = e^(-{lexp3f}×{u_fmt}) = {p_actual:.6f}\n"
                f"Optimizado: P(X>{u_fmt}) = e^(-{lambda_opt:.1f}×{u_fmt}) = {p_opt:.6f}"
            ),
            "desarrollo": [
                (f"Estado actual (λ = {lexp3f}):",
                 f"P(X>{u_fmt}) = {p_actual*100:.4f}%  →  {p_actual*10000:.0f} transacciones críticas por cada 10,000"),
                (f"Estado optimizado (λ = {lambda_opt:.1f}):",
                 f"P(X>{u_fmt}) = {p_opt*100:.4f}%  →  {p_opt*10000:.0f} transacciones críticas por cada 10,000"),
                ("Reducción absoluta de riesgo:",
                 f"Δ = {p_actual*100:.4f}% − {p_opt*100:.4f}% = {(p_actual-p_opt)*100:.4f} puntos porcentuales"),
                ("Reducción relativa (fórmula):",
                 f"({p_actual:.4f} − {p_opt:.4f}) / {p_actual:.4f} × 100 = {reduccion:.2f}%"),
            ],
            "conclusion": f"🚀 Pasar de λ={lexp3f} a λ={lambda_opt:.1f} reduce el riesgo en {reduccion:.1f}%, rescatando {(p_actual-p_opt)*10000:.0f} transacciones por cada 10,000 — con el umbral fijado en {u_fmt}s.",
            "color": COLORES['coral']
        },
    ]

    if modo == "📖 Paso a paso":
        paso_actual = st.slider("Selecciona el paso:", 1, len(pasos), 1) - 1
        paso = pasos[paso_actual]

        # Navegación
        col_prev, col_ind, col_next = st.columns([1, 3, 1])
        with col_prev:
            if paso_actual > 0:
                if st.button("← Anterior"):
                    st.session_state['paso'] = paso_actual - 1
        with col_ind:
            st.markdown(f"<p style='text-align:center;color:{COLORES['gray']};'>Paso {paso_actual+1} de {len(pasos)}</p>", unsafe_allow_html=True)
        with col_next:
            if paso_actual < len(pasos) - 1:
                if st.button("Siguiente →"):
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
            col_d, col_f = st.columns([1, 1])
            with col_d:
                st.markdown(f'<div class="explanation"><b>{i+1}.</b> {desc}</div>', unsafe_allow_html=True)
            with col_f:
                st.markdown(f'<div class="formula-block" style="font-size:0.95rem;">{form}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="alert-success">💡 <b>Conclusión:</b> {paso["conclusion"]}</div>', unsafe_allow_html=True)

    else:  # Desarrollo completo
        for i, paso in enumerate(pasos):
            with st.expander(f"{'✅' if i < 4 else '🚀'} {paso['titulo']}", expanded=(i == 0)):
                st.markdown(f'<div class="explanation">{paso["concepto"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="formula-block">{paso["formula"]}</div>', unsafe_allow_html=True)
                for desc, form in paso['desarrollo']:
                    c1, c2 = st.columns(2)
                    with c1: st.markdown(f'<div class="explanation">{desc}</div>', unsafe_allow_html=True)
                    with c2: st.markdown(f'<div class="formula-block" style="font-size:0.9rem;">{form}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="alert-success">💡 {paso["conclusion"]}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — SIMULADOR MONTE CARLO
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""
    <div class="tab-intro">
    🎲 <b>Simulador Monte Carlo — ¿cómo funciona?</b><br>
    El método Monte Carlo consiste en generar miles de números aleatorios que siguen la distribución
    del modelo (en este caso, exponencial con parámetro λ), y luego contar qué fracción supera el umbral.
    Si el modelo es correcto, esa fracción debe converger al valor analítico <b>e⁻ˡˣ</b>.<br><br>
    Esto ilustra la <b>Ley de los Grandes Números</b>: a medida que N crece, la probabilidad empírica
    se acerca cada vez más al valor teórico. La gráfica de convergencia muestra ese proceso en tiempo real.
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.markdown('<div class="section-header">⚙️ Parámetros</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation">
        <b>N simulaciones:</b> cuántos tiempos de respuesta aleatorios se generan.
        Más N = mayor precisión pero más tiempo de cómputo.<br><br>
        <b>λ:</b> tasa del modelo exponencial. Por defecto usa el λ estimado de los datos reales.
        Cámbialo para explorar cómo diferentes sistemas se comportarían.<br><br>
        <b>Semilla aleatoria:</b> fija la semilla del generador para obtener resultados reproducibles.
        </div>""", unsafe_allow_html=True)
        n_sim   = st.select_slider("Número de simulaciones", [100, 500, 1000, 5000, 10000, 50000], value=1000)
        lam_mc  = st.slider("λ para la simulación", 0.3, 4.0, float(round(lam_est, 2)), 0.1)
        seed_mc = st.number_input("Semilla aleatoria", 0, 9999, 42)

        p_analitica = prob_mayor(umbral, lam_mc)

        st.markdown(f"""
        <div class="alert-info">
        📐 <b>Valor analítico esperado:</b><br>
        P(X > {umbral:.0f}) = e⁻{umbral:.0f}×{lam_mc:.2f}<br>
        = <b>{p_analitica*100:.4f}%</b>
        </div>""", unsafe_allow_html=True)

        ejecutar = st.button("🚀 Ejecutar simulación", use_container_width=True, type="primary")

    with col_viz:
        if ejecutar or 'mc_resultados' in st.session_state:
            if ejecutar:
                np.random.seed(seed_mc)
                muestras = np.random.exponential(1/lam_mc, n_sim)
                criticos_acum = np.cumsum(muestras > umbral)
                n_range = np.arange(1, n_sim + 1)
                p_empirica_acum = criticos_acum / n_range
                st.session_state['mc_resultados'] = {
                    'muestras': muestras, 'p_acum': p_empirica_acum,
                    'n_range': n_range, 'p_analitica': p_analitica,
                    'lam': lam_mc, 'n': n_sim
                }

            res = st.session_state['mc_resultados']

            fig_mc = make_subplots(rows=1, cols=2,
                subplot_titles=["Convergencia de la probabilidad empírica", "Distribución de la muestra"])

            # Convergencia
            fig_mc.add_trace(go.Scatter(
                x=res['n_range'], y=res['p_acum'],
                mode='lines', line=dict(color=COLORES['navy'], width=1.5),
                name='P empírica acumulada'
            ), row=1, col=1)
            fig_mc.add_hline(
                y=res['p_analitica'],
                line=dict(color=COLORES['coral'], dash='dash', width=2),
                row=1, col=1
            )
            fig_mc.add_annotation(
                xref='x', yref='y',
                x=res['n'] * 0.7, y=res['p_analitica'] * 1.15,
                text=f"Valor analítico: {res['p_analitica']*100:.3f}%",
                font=dict(color=COLORES['coral'], size=11),
                showarrow=False, row=1, col=1
            )

            # Histograma
            fig_mc.add_trace(go.Histogram(
                x=res['muestras'], nbinsx=40,
                histnorm='probability density',
                marker=dict(color='rgba(6,90,130,0.5)', line=dict(color=COLORES['navy'], width=0.3)),
                name='Muestras'
            ), row=1, col=2)
            x_fit = np.linspace(0, min(res['muestras'].max() * 0.9, 15), 300)
            fig_mc.add_trace(go.Scatter(
                x=x_fit, y=f_exp(x_fit, res['lam']),
                mode='lines', line=dict(color=COLORES['coral'], width=2.5),
                name='f(x) teórica'
            ), row=1, col=2)
            fig_mc.add_vline(x=umbral, line=dict(color='#333', dash='dash'), row=1, col=2)

            fig_mc.update_layout(
                height=380, plot_bgcolor='#FAFAFA', paper_bgcolor='white',
                showlegend=False, margin=dict(l=40, r=20, t=50, b=40)
            )
            fig_mc.update_xaxes(title_text="N simulaciones", row=1, col=1)
            fig_mc.update_xaxes(title_text="Tiempo (s)", row=1, col=2)
            fig_mc.update_yaxes(title_text="P empírica", row=1, col=1)
            st.plotly_chart(fig_mc, use_container_width=True)

            # Resultados
            p_emp_final = res['p_acum'][-1]
            error_abs   = abs(p_emp_final - res['p_analitica'])
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.markdown(f"""
                <div class="kpi-card red">
                    <div class="kpi-label">P empírica final</div>
                    <div class="kpi-value">{p_emp_final*100:.3f}%</div>
                    <div class="kpi-sub">{int(np.sum(res['muestras'] > umbral))} eventos críticos en {res['n']:,}</div>
                </div>""", unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div class="kpi-card green">
                    <div class="kpi-label">Valor analítico</div>
                    <div class="kpi-value">{res['p_analitica']*100:.3f}%</div>
                    <div class="kpi-sub">e⁻{umbral:.0f}×{res['lam']:.2f}</div>
                </div>""", unsafe_allow_html=True)
            with col_r3:
                st.markdown(f"""
                <div class="kpi-card {'green' if error_abs < 0.005 else 'gold'}">
                    <div class="kpi-label">Error absoluto</div>
                    <div class="kpi-value">{error_abs*100:.4f}%</div>
                    <div class="kpi-sub">{'✅ Convergencia excelente' if error_abs < 0.005 else '⚠️ Aumenta N para mejorar'}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:3rem; color:#94A3B8;">
                <div style="font-size:3rem;">🎲</div>
                <p>Configura los parámetros y presiona <b>Ejecutar simulación</b></p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARACIÓN DE MODELOS
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div class="tab-intro">
    ⚖️ <b>¿Por qué elegimos la distribución exponencial y no otra?</b><br>
    Esta pestaña compara tres distribuciones ajustadas a los mismos datos: <b>Exponencial</b>, <b>Normal</b> y <b>Gamma</b>.
    La gráfica muestra cuál se ajusta mejor al histograma real de tiempos de respuesta, y la tabla compara
    sus probabilidades críticas.<br><br>
    La elección del modelo no es arbitraria — tiene un fundamento teórico sólido basado en la naturaleza
    de los sistemas de procesamiento de solicitudes, explicado en el panel derecho.
    </div>""", unsafe_allow_html=True)

    # Ajuste de parámetros
    mu_datos  = np.mean(datos[datos < np.percentile(datos, 95)])
    std_datos = np.std(datos[datos < np.percentile(datos, 95)])
    a_gamma   = (mu_datos / std_datos) ** 2
    b_gamma   = std_datos ** 2 / mu_datos

    p_exp_m  = prob_mayor(umbral, lam_est)
    p_norm_m = 1 - norm.cdf(umbral, mu_datos, std_datos)
    p_gam_m  = 1 - gamma.cdf(umbral, a_gamma, scale=b_gamma)

    col_g, col_t = st.columns([3, 1])

    with col_g:
        x = np.linspace(0, min(max(datos) * 0.85, 12), 400)

        fig_comp = go.Figure()
        # Histograma
        fig_comp.add_trace(go.Histogram(
            x=datos, nbinsx=40, histnorm='probability density',
            marker=dict(color='rgba(100,116,139,0.25)', line=dict(color='#94A3B8', width=0.5)),
            name='Datos observados'
        ))
        # Tres modelos
        modelos = [
            (f_exp(x, lam_est),          COLORES['navy'],  f'Exponencial (λ={lam_est:.3f})',        p_exp_m),
            (f_norm(x, mu_datos, std_datos), COLORES['gold'],  f'Normal (μ={mu_datos:.2f}, σ={std_datos:.2f})', p_norm_m),
            (f_gamma(x, a_gamma, b_gamma),   COLORES['purple'], f'Gamma (α={a_gamma:.2f}, β={b_gamma:.2f})',    p_gam_m),
        ]
        for y_m, color, name, p_m in modelos:
            fig_comp.add_trace(go.Scatter(
                x=x, y=y_m, mode='lines',
                line=dict(color=color, width=2.5),
                name=f'{name} → P(X>{umbral:.0f}s)={p_m*100:.2f}%'
            ))
        fig_comp.add_vline(x=umbral, line=dict(color='#333', dash='dash', width=2))
        fig_comp.update_layout(
            title="Comparación de distribuciones ajustadas a los datos",
            xaxis_title="Tiempo de respuesta (s)", yaxis_title="Densidad",
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=400, margin=dict(l=50, r=20, t=70, b=50)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_t:
        st.markdown('<div class="section-header">📊 Comparativa</div>', unsafe_allow_html=True)
        tabla_modelos = pd.DataFrame({
            'Modelo': ['Exponencial', 'Normal', 'Gamma'],
            f'P(X>{umbral:.0f}s)': [f'{p_exp_m*100:.3f}%', f'{p_norm_m*100:.3f}%', f'{p_gam_m*100:.3f}%'],
            'E[X]': [f'{E_actual:.3f}s', f'{mu_datos:.3f}s', f'{a_gamma*b_gamma:.3f}s'],
        })
        st.dataframe(tabla_modelos, hide_index=True, use_container_width=True)

        st.markdown("""
        <div class="alert-success">
        ✅ <b>¿Por qué exponencial?</b><br><br>
        • Los tiempos de servicio en APIs siguen procesos de Poisson, cuyo tiempo entre eventos sigue una distribución exponencial.<br><br>
        • La exponencial tiene la propiedad de <b>falta de memoria</b>: el tiempo restante no depende del tiempo ya transcurrido.<br><br>
        • Es el modelo estándar en <b>teoría de colas</b> y análisis de sistemas de red.<br><br>
        • Tiene un solo parámetro (λ), lo que simplifica la estimación y la interpretación.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-info">
        📋 <b>Limitación</b><br>
        La Normal puede tomar valores negativos (tiempo < 0), lo que no tiene sentido físico para tiempos de respuesta.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — IMPACTO DE NEGOCIO
# ══════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""
    <div class="tab-intro">
    💰 <b>Calculadora de impacto de negocio — ¿cómo funciona?</b><br>
    Esta sección traduce el resultado matemático abstracto (un porcentaje de probabilidad) en una cifra
    económica concreta. El modelo parte de la probabilidad de latencia crítica que calculamos con la integral,
    y la combina con datos operativos del negocio para estimar cuánto dinero se pierde por día por causa de
    la latencia — y cuánto se recuperaría al optimizar el sistema.<br><br>
    Ajusta las variables en el panel izquierdo para adaptarlo al contexto real de PayFast Colombia.
    </div>""", unsafe_allow_html=True)

    # ── Sección: ¿Qué significa optimizar? ───────────────────────────────────
    with st.expander("🔧 ¿Qué significa optimizar el sistema? — Entendiendo λ y su impacto", expanded=True):

        col_exp1, col_exp2 = st.columns([1, 1])

        with col_exp1:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:1.3rem 1.5rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:1rem;
                        border-left:5px solid #065A82;color:#334155;">
            <b style="font-size:1rem;color:#021B2E;">¿Qué es λ y qué representa físicamente?</b><br><br>
            <span style="color:#334155;">λ (lambda) es el parámetro de la distribución exponencial que usamos para modelar los tiempos
            de respuesta de la API. Matemáticamente aparece en la función de densidad como:</span><br><br>
            <div style="background:linear-gradient(135deg,#EAF4FB,#F0FDF9);border:1.5px solid #AEDFF7;
                        border-radius:10px;padding:0.9rem 1.2rem;font-family:'Courier New',monospace;
                        font-size:0.95rem;color:#065A82;font-weight:700;text-align:center;margin:0.6rem 0;">
                f(x) = λ · e^(−λx)
            </div><br>
            <span style="color:#334155;">Su significado físico es directo: <b style="color:#021B2E;">λ es la tasa de procesamiento del sistema</b>,
            es decir, cuántas solicitudes por segundo puede atender la API.<br><br>
            De ahí se derivan dos relaciones clave:<br><br>
            &bull; <b style="color:#021B2E;">Tiempo promedio de respuesta:</b> E[X] = 1/λ<br>
            &bull; <b style="color:#021B2E;">Probabilidad de latencia crítica:</b> P(X &gt; umbral) = e^(−λ × umbral)<br><br>
            λ y el tiempo promedio son <b style="color:#021B2E;">inversamente proporcionales</b>:
            duplicar λ divide el tiempo promedio a la mitad.</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:1.3rem 1.5rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:1rem;
                        border-left:5px solid #1C7293;color:#334155;">
            <b style="font-size:1rem;color:#021B2E;">¿Qué pasa matemáticamente cuando λ aumenta?</b><br><br>
            <span style="color:#334155;">La función P(X &gt; {umbral:g}) = e^(−λ × {umbral:g}) es
            <b style="color:#021B2E;">decreciente en λ</b>. Cualquier mejora que incremente λ
            reduce exponencialmente el riesgo:</span><br><br>
            <div style="background:linear-gradient(135deg,#EAF4FB,#F0FDF9);border:1.5px solid #AEDFF7;
                        border-radius:10px;padding:0.9rem 1.2rem;font-family:'Courier New',monospace;
                        font-size:0.88rem;color:#065A82;font-weight:700;text-align:center;margin:0.6rem 0;">
                λ = {lam_est:.3f} → P = e^(−{lam_est:.3f}×{umbral:g}) = {p_actual:.4f} ({p_actual*100:.2f}%)<br>
                λ = {lambda_opt:.1f} &nbsp;&nbsp; → P = e^(−{lambda_opt:.1f}×{umbral:g}) = {p_opt:.4f} ({p_opt*100:.3f}%)<br>
                Reducción: {reduccion:.1f}% menos riesgo
            </div><br>
            <span style="color:#334155;">La curva verde en la pestaña <b style="color:#021B2E;">Densidad y Probabilidad</b> muestra
            visualmente cómo la distribución se desplaza hacia la izquierda al aumentar λ,
            concentrando más probabilidad en tiempos bajos.</span>
            </div>""", unsafe_allow_html=True)

        with col_exp2:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:1.3rem 1.5rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:1rem;
                        border-left:5px solid #02C39A;color:#334155;">
            <b style="font-size:1rem;color:#021B2E;">¿Qué acciones reales aumentan λ?</b><br><br>
            <span style="color:#475569;">Aumentar λ equivale a
            <b style="color:#021B2E;">reducir el tiempo promedio de respuesta</b> de la API.
            Estas son las estrategias más comunes:</span><br><br>

            <div style="margin-bottom:0.8rem;">
            <b style="color:#065A82;">⚡ Escalabilidad horizontal</b><br>
            <span style="color:#475569;font-size:0.88rem;">Añadir más instancias del servidor distribuye la carga.
            Pasar de 1 a 2 servidores idénticos duplica la tasa efectiva (λ × 2).</span>
            </div>

            <div style="margin-bottom:0.8rem;">
            <b style="color:#065A82;">🗄️ Optimización de base de datos</b><br>
            <span style="color:#475569;font-size:0.88rem;">Gran parte de la latencia proviene de consultas a BD.
            Añadir índices, usar caché (Redis) o migrar a una BD más rápida puede reducir el tiempo
            a la mitad — equivalente a duplicar λ.</span>
            </div>

            <div style="margin-bottom:0.8rem;">
            <b style="color:#065A82;">🌐 CDN y balanceo de carga</b><br>
            <span style="color:#475569;font-size:0.88rem;">Un balanceador distribuye solicitudes entre servidores,
            evitando saturación. Reduce los picos de latencia y sube el λ efectivo.</span>
            </div>

            <div style="margin-bottom:0.8rem;">
            <b style="color:#065A82;">🔧 Optimización del código</b><br>
            <span style="color:#475569;font-size:0.88rem;">Eliminar llamadas síncronas, usar procesamiento asíncrono
            o reducir el payload de respuesta disminuye directamente el tiempo por solicitud.</span>
            </div>

            <div>
            <b style="color:#065A82;">☁️ Autoescalado en cloud</b><br>
            <span style="color:#475569;font-size:0.88rem;">AWS Lambda, Google Cloud Run y similares ajustan la
            capacidad automáticamente según la demanda, manteniendo λ alto en picos de tráfico.</span>
            </div>
            </div>""", unsafe_allow_html=True)

        # Tabla de sensibilidad: cómo cambia P según distintos valores de λ
        st.markdown(f"""
        <div style="font-size:1.1rem;font-weight:700;color:#021B2E;padding:0.5rem 0;
                    margin:1rem 0 0.3rem 0;">
            📉 Tabla de sensibilidad — efecto de λ sobre el riesgo
        </div>
        <div style="background:#F8FAFC;border-left:3px solid #02C39A;padding:0.7rem 1rem;
                    border-radius:0 8px 8px 0;font-size:0.88rem;color:#334155;margin-bottom:0.6rem;">
        Esta tabla muestra cómo varía P(X &gt; {umbral:g}s) al mejorar λ progresivamente.
        Cada fila es un escenario hipotético de optimización. El λ actual es <b style="color:#021B2E;">{lam_est:.3f}</b>.
        </div>""", unsafe_allow_html=True)

        lambdas_sens = [round(lam_est * f, 3) for f in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]]
        filas = []
        for l in lambdas_sens:
            p_l   = float(np.exp(-l * umbral))
            e_l   = 1.0 / l
            red_l = (p_actual - p_l) / p_actual * 100 if p_actual > 0 else 0
            escenario = "← estado actual" if abs(l - lam_est) < 0.01 else (
                        "← λ optimizado" if abs(l - lambda_opt) < 0.05 else "")
            filas.append({
                'λ': f'{l:.3f}',
                'E[X] = 1/λ (s)': f'{e_l:.3f}s',
                f'P(X>{umbral:g}s)': f'{p_l*100:.4f}%',
                'Reducción vs. actual': f'{red_l:.1f}%' if red_l > 0 else '—',
                'Escenario': escenario
            })
        df_sens = pd.DataFrame(filas)
        st.dataframe(df_sens, hide_index=True, use_container_width=True)

        # Gráfica de sensibilidad
        l_range = np.linspace(max(0.1, lam_est * 0.5), lam_est * 4, 200)
        p_range = np.exp(-l_range * umbral)
        e_range = 1.0 / l_range

        fig_sens = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"P(X > {umbral:g}s) en función de λ",
                "Tiempo promedio E[X] en función de λ"
            ]
        )
        fig_sens.add_trace(go.Scatter(
            x=l_range, y=p_range * 100,
            mode='lines', line=dict(color=COLORES['coral'], width=2.5),
            name='P(X > umbral) %'
        ), row=1, col=1)
        # Marca el estado actual
        fig_sens.add_trace(go.Scatter(
            x=[lam_est], y=[p_actual * 100],
            mode='markers', marker=dict(color=COLORES['navy'], size=10, symbol='circle'),
            name=f'Estado actual (λ={lam_est:.3f})'
        ), row=1, col=1)
        # Marca el estado optimizado
        fig_sens.add_trace(go.Scatter(
            x=[lambda_opt], y=[p_opt * 100],
            mode='markers', marker=dict(color=COLORES['mint'], size=10, symbol='star'),
            name=f'Optimizado (λ={lambda_opt:.1f})'
        ), row=1, col=1)
        fig_sens.add_trace(go.Scatter(
            x=l_range, y=e_range,
            mode='lines', line=dict(color=COLORES['navy'], width=2.5),
            name='E[X] (s)'
        ), row=1, col=2)
        fig_sens.add_trace(go.Scatter(
            x=[lam_est], y=[E_actual],
            mode='markers', marker=dict(color=COLORES['coral'], size=10, symbol='circle'),
            name=f'Actual: {E_actual:.3f}s'
        ), row=1, col=2)
        fig_sens.add_trace(go.Scatter(
            x=[lambda_opt], y=[E_opt],
            mode='markers', marker=dict(color=COLORES['mint'], size=10, symbol='star'),
            name=f'Optimizado: {E_opt:.3f}s'
        ), row=1, col=2)

        fig_sens.add_vline(x=lam_est,    line=dict(color=COLORES['navy'], dash='dot', width=1.5), row=1, col=1)
        fig_sens.add_vline(x=lambda_opt, line=dict(color=COLORES['mint'], dash='dot', width=1.5), row=1, col=1)
        fig_sens.add_vline(x=lam_est,    line=dict(color=COLORES['navy'], dash='dot', width=1.5), row=1, col=2)
        fig_sens.add_vline(x=lambda_opt, line=dict(color=COLORES['mint'], dash='dot', width=1.5), row=1, col=2)

        fig_sens.update_xaxes(title_text="λ (tasa de procesamiento)", row=1, col=1)
        fig_sens.update_xaxes(title_text="λ (tasa de procesamiento)", row=1, col=2)
        fig_sens.update_yaxes(title_text=f"P(X>{umbral:g}s) [%]", row=1, col=1)
        fig_sens.update_yaxes(title_text="E[X] [segundos]", row=1, col=2)
        fig_sens.update_layout(
            height=360, plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            showlegend=False, margin=dict(l=50, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#EFF6FF,#DBEAFE);border:1px solid #93C5FD;
                    border-radius:10px;padding:0.9rem 1.1rem;color:#1E40AF;font-size:0.88rem;margin:0.5rem 0;">
        📌 <b>Cómo leer estas gráficas:</b> el punto <b>azul</b> es el estado actual (λ = {lam_est:.3f}),
        el punto <b>verde ★</b> es el escenario optimizado (λ = {lambda_opt:.1f}) configurado en el panel izquierdo.
        Ambas curvas son decrecientes: mayor λ = menor riesgo y menor tiempo promedio.
        El descenso es <b>exponencial</b>, no lineal — pequeñas mejoras en λ cerca del valor actual
        tienen mayor impacto relativo que las mismas mejoras en valores altos.
        </div>""", unsafe_allow_html=True)

    # ── Cálculos ──────────────────────────────────────────────────────────────
    trans_criticas_dia     = transacciones_dia * p_actual
    trans_criticas_opt_dia = transacciones_dia * p_opt
    trans_perdidas_dia     = trans_criticas_dia * tasa_abandono
    trans_perdidas_opt_dia = trans_criticas_opt_dia * tasa_abandono
    trans_rescatadas_dia   = trans_perdidas_dia - trans_perdidas_opt_dia
    perdida_dia      = trans_perdidas_dia * valor_promedio
    perdida_opt_dia  = trans_perdidas_opt_dia * valor_promedio
    ahorro_dia       = perdida_dia - perdida_opt_dia
    ahorro_mes       = ahorro_dia * 30
    ahorro_anio      = ahorro_dia * 365

    # ── Explicación del modelo de cálculo ──────────────────────────────────────
    with st.expander("📐 ¿Cómo se calculan estas cifras? — Modelo matemático paso a paso", expanded=True):
        st.markdown("""
        <div class="explanation">
        El modelo de impacto económico encadena cuatro pasos. Cada uno depende del anterior,
        formando una cadena que va desde la probabilidad matemática hasta el peso colombiano:
        </div>""", unsafe_allow_html=True)

        pasos_neg = [
            (
                "1️⃣  Transacciones críticas por día",
                "Del total de transacciones diarias, ¿cuántas tienen latencia superior al umbral?",
                f"Trans. críticas = Transacciones/día × P(X > {umbral:.0f}s)",
                f"= {transacciones_dia:,} × {p_actual:.4f} = {trans_criticas_dia:,.1f} transacciones/día",
                "La probabilidad P(X > umbral) viene directamente de la integral resuelta: e⁻ˡˣ"
            ),
            (
                "2️⃣  Transacciones perdidas por abandono",
                "No todas las transacciones lentas se pierden — algunos usuarios esperan. La tasa de abandono modela qué fracción se va.",
                f"Trans. perdidas = Trans. críticas × Tasa de abandono",
                f"= {trans_criticas_dia:,.1f} × {tasa_abandono:.2f} = {trans_perdidas_dia:,.1f} transacciones/día",
                "La tasa de abandono del 65% por defecto proviene de estudios de UX que muestran que más del 60% de los usuarios abandona si la espera supera 3 segundos."
            ),
            (
                "3️⃣  Pérdida económica diaria",
                "Cada transacción perdida representa un ingreso que no se procesó. Multiplicamos por el valor promedio de cada transacción.",
                f"Pérdida diaria = Trans. perdidas × Valor promedio",
                f"= {trans_perdidas_dia:,.1f} × ${valor_promedio:,} = ${perdida_dia:,.0f} COP/día",
                "Este es el costo directo de la latencia en términos de ventas no realizadas."
            ),
            (
                "4️⃣  Ahorro por optimización",
                "Si mejoramos el sistema (aumentamos λ), la nueva probabilidad de latencia es menor. La diferencia entre la pérdida actual y la nueva es el ahorro.",
                f"Ahorro = Pérdida actual − Pérdida optimizada",
                f"= ${perdida_dia:,.0f} − ${perdida_opt_dia:,.0f} = ${ahorro_dia:,.0f} COP/día → ${ahorro_anio:,.0f} COP/año",
                f"Aumentar λ de {lam_est:.3f} a {lambda_opt:.1f} reduce P(X>{umbral:.0f}s) de {p_actual*100:.2f}% a {p_opt*100:.3f}%, lo que se traduce directamente en menos pérdidas."
            ),
        ]

        for titulo, concepto, formula, resultado, nota in pasos_neg:
            st.markdown(f"<div style='margin-top:1rem;'><b style='font-size:1rem;color:#021B2E;'>{titulo}</b></div>", unsafe_allow_html=True)
            ca, cb = st.columns([1, 1])
            with ca:
                st.markdown(f'<div class="explanation"><b>¿Qué calcula?</b><br>{concepto}<br><br><i>💡 {nota}</i></div>', unsafe_allow_html=True)
            with cb:
                st.markdown(f'<div class="formula-block" style="font-size:0.9rem;">{formula}<br><br>{resultado}</div>', unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    metricas_neg = [
        (col_m1, "red",    "Pérdida diaria actual",    f"${perdida_dia:,.0f}",       "COP / día"),
        (col_m2, "green",  "Pérdida tras optimización", f"${perdida_opt_dia:,.0f}",   "COP / día"),
        (col_m3, "gold",   "Ahorro mensual",            f"${ahorro_mes:,.0f}",        "COP / mes"),
        (col_m4, "purple", "Ahorro anual",              f"${ahorro_anio:,.0f}",       "COP / año"),
    ]
    for col, cls, label, val, sub in metricas_neg:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="font-size:1.4rem;">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        categorias = ['Trans. críticas/día', 'Trans. perdidas/día', 'Pérdida diaria\n(miles COP)']
        antes   = [trans_criticas_dia, trans_perdidas_dia, perdida_dia/1000]
        despues = [trans_criticas_opt_dia, trans_perdidas_opt_dia, perdida_opt_dia/1000]

        fig_neg = go.Figure()
        fig_neg.add_trace(go.Bar(
            name=f'Estado actual (λ={lam_est:.3f})', x=categorias, y=antes,
            marker_color=COLORES['coral'],
            text=[f'{v:,.1f}' for v in antes], textposition='outside'
        ))
        fig_neg.add_trace(go.Bar(
            name=f'Tras optimización (λ={lambda_opt:.1f})', x=categorias, y=despues,
            marker_color=COLORES['mint'],
            text=[f'{v:,.1f}' for v in despues], textposition='outside'
        ))
        fig_neg.update_layout(
            barmode='group',
            title=f"Impacto operativo: antes (λ={lam_est:.3f}) vs. después (λ={lambda_opt:.1f})",
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            height=400, margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_neg, use_container_width=True)

        st.markdown(f"""
        <div class="alert-info">
        📌 <b>Cómo leer esta gráfica:</b> cada grupo de barras compara el mismo indicador antes y después
        de la optimización. Las barras rojas son el estado actual del sistema (λ = {lam_est:.3f}),
        las verdes son el escenario optimizado (λ = {lambda_opt:.1f}).
        La diferencia entre ellas es el impacto directo de mejorar la infraestructura.
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">📋 Desglose detallado</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="explanation">
        Esta tabla muestra cada variable del modelo con su valor numérico. Sirve como trazabilidad
        completa: desde la probabilidad matemática hasta la cifra financiera final, con cada
        paso intermedio visible y verificable.
        </div>""", unsafe_allow_html=True)
        desglose = pd.DataFrame({
            'Concepto': [
                'Transacciones/día',
                f'P(X>{umbral:.0f}s) actual  [integral]',
                f'P(X>{umbral:.0f}s) optimizado  [integral]',
                'Trans. críticas/día (actual)',
                'Trans. críticas/día (opt.)',
                'Tasa de abandono  [parámetro]',
                'Trans. perdidas/día (actual)',
                'Trans. perdidas/día (opt.)',
                'Trans. rescatadas/día',
                'Valor promedio por transacción',
                'Pérdida diaria actual',
                'Pérdida diaria optimizada',
                'Ahorro diario',
                'Ahorro mensual (×30)',
                'Ahorro anual (×365)',
            ],
            'Valor': [
                f'{transacciones_dia:,}',
                f'{p_actual*100:.4f}%  = e⁻{umbral:.0f}×{lam_est:.3f}',
                f'{p_opt*100:.4f}%  = e⁻{umbral:.0f}×{lambda_opt:.1f}',
                f'{trans_criticas_dia:,.1f}',
                f'{trans_criticas_opt_dia:,.1f}',
                f'{tasa_abandono*100:.0f}%',
                f'{trans_perdidas_dia:,.1f}',
                f'{trans_perdidas_opt_dia:,.1f}',
                f'{trans_rescatadas_dia:,.1f}',
                f'${valor_promedio:,} COP',
                f'${perdida_dia:,.0f} COP',
                f'${perdida_opt_dia:,.0f} COP',
                f'${ahorro_dia:,.0f} COP',
                f'${ahorro_mes:,.0f} COP',
                f'${ahorro_anio:,.0f} COP',
            ]
        })
        st.dataframe(desglose, hide_index=True, use_container_width=True, height=480)

    # ── Origen de los datos y supuestos ───────────────────────────────────────
    st.markdown("<div class='section-header'>📌 Origen de los datos y supuestos del modelo</div>", unsafe_allow_html=True)

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        st.markdown(f"""
        <div class="step-card">
        <b>🔵 Parámetros derivados del modelo matemático</b><br><br>
        Estos valores <b>no son inventados ni configurados manualmente</b> — se calculan
        automáticamente a partir de los datos de latencia cargados en la app, usando la
        distribución exponencial ajustada con el estimador MLE (λ = 1/x̄):<br><br>
        • <b>P(X &gt; {umbral:g}s) actual = {p_actual*100:.4f}%</b> — resultado directo de la integral
        e<sup>−λ·{umbral:g}</sup> con λ = {lam_est:.4f} estimado de los {'datos cargados' if uploaded else 'datos simulados'}.<br><br>
        • <b>P(X &gt; {umbral:g}s) optimizado = {p_opt*100:.4f}%</b> — mismo cálculo con el λ optimizado
        que se ajusta en el panel izquierdo (representa un escenario hipotético de mejora).<br><br>
        • <b>Transacciones críticas/día</b> — producto directo de la probabilidad × transacciones configuradas.
        </div>""", unsafe_allow_html=True)

    with col_o2:
        origen_datos = "datos reales cargados por el usuario (CSV)" if uploaded else "datos <b>simulados artificialmente</b> (500 muestras exponencial(λ=1) + 15 outliers uniformes entre 4–8s, generados con NumPy)"
        st.markdown(f"""
        <div class="step-card">
        <b>🟡 Parámetros configurables (valores de referencia)</b><br><br>
        Estos valores <b>no provienen de datos reales de PayFast</b> — son parámetros de
        entrada que el usuario ajusta en el panel izquierdo para simular distintos escenarios
        de negocio. Los valores por defecto son estimaciones de referencia académica:<br><br>
        • <b>Transacciones/día (10,000):</b> volumen hipotético representativo de una fintech colombiana mediana.<br><br>
        • <b>Valor promedio por transacción ($150,000 COP):</b> estimación genérica basada en el ticket
        promedio de pagos electrónicos en Colombia (Superintendencia Financiera, 2023).<br><br>
        • <b>Tasa de abandono (65%):</b> tomada de estudios de UX que reportan que entre el 53% y 70%
        de usuarios abandona una transacción si la espera supera 3 segundos (Google/SOASTA, 2017).<br><br>
        📂 <b>Fuente de los tiempos de respuesta:</b> {origen_datos}.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-danger" style="margin-top:0.5rem;">
    ⚠️ <b>Importante:</b> Las cifras económicas de esta sección son <b>proyecciones ilustrativas</b>
    construidas sobre parámetros configurables, no sobre registros financieros reales de PayFast Colombia.
    Su propósito es demostrar cómo el resultado matemático (la probabilidad de latencia) se conecta con
    consecuencias operativas concretas. Para un análisis real se requeriría datos históricos de
    transacciones, tasas de abandono propias y valores de ticket reales de la empresa.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — ANÁLISIS DE DATOS
# ══════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("""
    <div class="tab-intro">
    📊 <b>Exploración y validación de los datos.</b><br>
    Antes de aplicar cualquier modelo probabilístico, es necesario explorar los datos para entender
    su comportamiento. Esta pestaña responde a: ¿los datos observados son consistentes con el modelo
    exponencial que elegimos?<br><br>
    El <b>histograma</b> muestra la distribución real de los tiempos de respuesta registrados.
    La <b>curva roja</b> es la distribución exponencial ajustada con el λ estimado por MLE.
    Si la curva se superpone bien al histograma, el modelo es una buena representación del sistema.
    </div>""", unsafe_allow_html=True)

    col_hist, col_stats = st.columns([3, 2])

    with col_hist:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=datos, nbinsx=45, histnorm='probability density',
            marker=dict(color='rgba(6,90,130,0.45)', line=dict(color=COLORES['navy'], width=0.4)),
            name='Datos observados'
        ))
        x_fit = np.linspace(0, min(np.max(datos)*0.9, 14), 300)
        fig_h.add_trace(go.Scatter(
            x=x_fit, y=f_exp(x_fit, lam_est),
            mode='lines', line=dict(color=COLORES['coral'], width=2.5),
            name=f'Exp ajustada (λ={lam_est:.3f})'
        ))
        fig_h.add_vline(x=umbral, line=dict(color='#333', dash='dash', width=2))
        fig_h.add_annotation(x=umbral+0.1, y=lam_est*0.6,
            text=f"umbral = {umbral:.0f}s", showarrow=False,
            font=dict(size=10, color='#333'), bgcolor='rgba(255,255,255,0.8)')
        fig_h.update_layout(
            title="Histograma de tiempos de respuesta vs. distribución ajustada",
            xaxis_title="Tiempo (s)", yaxis_title="Densidad",
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=380, margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with col_stats:
        st.markdown('<div class="section-header">📐 Estadísticas descriptivas</div>', unsafe_allow_html=True)
        percentiles = [50, 75, 90, 95, 99]
        stats_df = pd.DataFrame({
            'Estadístico': [
                'N registros', 'Media (x̄)', 'Mediana', 'Desv. estándar',
                'Mínimo', 'Máximo',
                *[f'Percentil {p}' for p in percentiles],
                f'% > {umbral:.0f}s (empírico)',
                'λ estimado (MLE)'
            ],
            'Valor': [
                f'{len(datos):,}',
                f'{np.mean(datos):.4f} s',
                f'{np.median(datos):.4f} s',
                f'{np.std(datos):.4f} s',
                f'{np.min(datos):.4f} s',
                f'{np.max(datos):.4f} s',
                *[f'{np.percentile(datos, p):.4f} s' for p in percentiles],
                f'{np.mean(datos>umbral)*100:.2f}%',
                f'{lam_est:.4f}'
            ]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True, height=410)
        st.markdown(f"""
        <div class="explanation">
        <b>¿Qué indica λ = {lam_est:.4f}?</b><br>
        El estimador MLE (Máxima Verosimilitud) calcula λ = 1/x̄ = 1/{np.mean(datos):.4f} = {lam_est:.4f}.
        Esto significa que en promedio el sistema procesa {lam_est:.2f} transacciones por segundo,
        con un tiempo medio de respuesta de {E_actual:.3f} segundos.
        </div>""", unsafe_allow_html=True)

    # Serie de tiempo
    if 'timestamp' in df.columns:
        st.markdown('<div class="section-header">⏱️ Serie de tiempo</div>', unsafe_allow_html=True)
        df_plot = df.copy()
        df_plot['critico'] = df_plot['response_time_s'] > umbral
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot['response_time_s'],
            mode='lines', line=dict(color=COLORES['navy'], width=1),
            name='Tiempo de respuesta'
        ))
        criticos = df_plot[df_plot['critico']]
        fig_ts.add_trace(go.Scatter(
            x=criticos['timestamp'], y=criticos['response_time_s'],
            mode='markers', marker=dict(color=COLORES['coral'], size=7, symbol='circle'),
            name=f'Latencia crítica (>{umbral:.0f}s)'
        ))
        fig_ts.add_hline(y=umbral, line=dict(color=COLORES['coral'], dash='dash', width=1.5))
        fig_ts.update_layout(
            xaxis_title="Timestamp", yaxis_title="Tiempo (s)",
            plot_bgcolor='#FAFAFA', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=300, margin=dict(l=50, r=20, t=30, b=50)
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        st.markdown(f"""
        <div class="explanation">
        Los <b>puntos rojos</b> son los registros donde el tiempo de respuesta superó el umbral de {umbral:.0f}s.
        En una API bien configurada estos puntos deben ser escasos y aleatorios — sin patrones de agrupamiento
        que indiquen problemas sistemáticos (picos de carga, cuellos de botella recurrentes, etc.).
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 7 — INTERVALO PERSONALIZADO
# ══════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("""
    <div class="tab-intro">
    📐 <b>Calculadora de probabilidad en cualquier intervalo [a, b].</b><br>
    Una de las aplicaciones más útiles de la integral definida es calcular probabilidades en
    intervalos arbitrarios. En lugar de preguntar solo "¿cuántas transacciones tardan más de 3s?",
    podemos preguntar cosas como: "¿qué fracción de transacciones tarda entre 1 y 2 segundos?"
    o "¿cuántas caen en la zona óptima de 0 a 1 segundo?"<br><br>
    La gráfica muestra el área de tu intervalo sombreada en amarillo.
    Ajusta <b>a</b> y <b>b</b> en el panel izquierdo para explorar cualquier rango.
    </div>""", unsafe_allow_html=True)

    if a_custom >= b_custom:
        st.warning("⚠️ El límite inferior debe ser menor que el superior.")
    else:
        p_int     = prob_intervalo(a_custom, b_custom, f_exp, lam_est)
        p_int_opt = prob_intervalo(a_custom, b_custom, f_exp, lambda_opt)

        col_g2, col_i2 = st.columns([3, 1])

        with col_g2:
            x_max2 = max(b_custom * 1.8, 8)
            x2 = np.linspace(0, x_max2, 500)

            # Tres zonas
            xa = x2[x2 <= a_custom]
            xi = x2[(x2 >= a_custom) & (x2 <= b_custom)]
            xb = x2[x2 >= b_custom]

            fig_int = go.Figure()
            for xz, label, color in [
                (xa, f'P(X < {a_custom:.1f}s)', 'rgba(2,195,154,0.15)'),
                (xi, f'P({a_custom:.1f} < X < {b_custom:.1f}s)', 'rgba(245,166,35,0.45)'),
                (xb, f'P(X > {b_custom:.1f}s)', 'rgba(249,97,103,0.2)')
            ]:
                if len(xz) > 1:
                    yz = f_exp(xz, lam_est)
                    fig_int.add_trace(go.Scatter(
                        x=np.concatenate([[xz[0]], xz, [xz[-1]]]),
                        y=np.concatenate([[0], yz, [0]]),
                        fill='toself', fillcolor=color,
                        line=dict(color='rgba(0,0,0,0)'), name=label
                    ))

            fig_int.add_trace(go.Scatter(
                x=x2, y=f_exp(x2, lam_est),
                mode='lines', line=dict(color=COLORES['navy'], width=3),
                name=f'f(x), λ={lam_est:.3f}'
            ))
            fig_int.add_vline(x=a_custom, line=dict(color=COLORES['gold'], dash='dash', width=2))
            fig_int.add_vline(x=b_custom, line=dict(color=COLORES['gold'], dash='dash', width=2))

            # Anotación del área
            x_mid2 = (a_custom + b_custom) / 2
            y_mid2 = f_exp(x_mid2, lam_est) * 0.5
            fig_int.add_annotation(
                x=x_mid2, y=y_mid2 + 0.08,
                text=f"<b>{p_int*100:.4f}%</b>",
                showarrow=True, arrowhead=2, arrowcolor=COLORES['gold'],
                font=dict(size=13, color=COLORES['gold']),
                bgcolor='rgba(255,255,255,0.9)', bordercolor=COLORES['gold'], borderwidth=1
            )
            fig_int.update_layout(
                title=f"P({a_custom:.1f} < X < {b_custom:.1f}) = área sombreada en amarillo",
                xaxis_title="Tiempo (s)", yaxis_title="f(x)",
                plot_bgcolor='#FAFAFA', paper_bgcolor='white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=400, margin=dict(l=50, r=20, t=60, b=50)
            )
            st.plotly_chart(fig_int, use_container_width=True)

        with col_i2:
            st.markdown('<div class="section-header">🧮 Cálculo</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="explanation">
            La integral en un intervalo cerrado [a, b] se resuelve usando el Teorema
            Fundamental del Cálculo: evaluamos la antiderivada en b y en a, y restamos.
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="formula-block">∫_{a_custom:.1f}^{b_custom:.1f} λe⁻ˡˣ dx<br><br>= [-e⁻ˡˣ]_{a_custom:.1f}^{b_custom:.1f}<br><br>= e⁻{a_custom*lam_est:.3f} − e⁻{b_custom*lam_est:.3f}<br><br>= <b>{p_int:.6f}</b></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="alert-success">
            📊 <b>Resultado</b><br>
            P = <b>{p_int*100:.4f}%</b><br>
            ≈ {p_int*len(datos):.0f} de {len(datos)} registros caen en este rango<br><br>
            <b>Con λ optimizado ({lambda_opt:.1f}):</b><br>
            P = <b>{p_int_opt*100:.4f}%</b>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">📋 Las tres zonas</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="explanation">
            La integral de f(x) en todo su dominio es 1 (100%). Las tres zonas coloreadas
            en la gráfica son porciones de ese total. Su suma debe dar ≈ 100%.
            </div>""", unsafe_allow_html=True)
            p_antes   = prob_intervalo(0, a_custom, f_exp, lam_est)
            p_despues = prob_intervalo(b_custom, 1000, f_exp, lam_est)
            tabla_z = pd.DataFrame({
                'Zona': [f'Verde: X < {a_custom:.1f}s', f'Amarilla: [{a_custom:.1f}, {b_custom:.1f}]s', f'Roja: X > {b_custom:.1f}s', 'Total'],
                '%': [f'{p_antes*100:.2f}%', f'{p_int*100:.4f}%', f'{p_despues*100:.2f}%', '≈ 100%']
            })
            st.dataframe(tabla_z, hide_index=True, use_container_width=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94A3B8; font-size:0.82rem; padding:0.5rem;'>
⚡ PayFast Colombia S.A.S. · Análisis Probabilístico de Latencia API ·
Fundación Universitaria Compensar · Cálculo Integral · 2026<br>
<span style='color:#02C39A;'>Modelo: f(x) = λe⁻ˡˣ · P(X>3) = e⁻³ˡ · E[X] = 1/λ</span>
</div>
""", unsafe_allow_html=True)
