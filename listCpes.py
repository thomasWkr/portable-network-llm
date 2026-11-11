import pandas as pd

# 1. Carrega o arquivo anterior com os dados já filtrados por fabricante
df = pd.read_csv('equipamentos_telecom_filtrados.csv', sep=',', low_memory=False)

# 2. Tipos oficiais exatos na Anatel para redes ópticas e roteadores estruturais
tipos_fibra_rede = [
    'ONT - Terminação de Rede Ótica',
    'ONU - Unidade de Rede Óptica',
    'Terminal de Linha Óptica',               # OLTs (Central)
    'Equipamento para Interconexão de Redes', # Roteadores de borda/core
    'Equipamento de Rede de Dados'            # Roteadores e switches de dados
]

# 3. Palavras-chave estritas para resgatar roteadores residenciais que 
# vêm cadastrados sob a categoria genérica de "Transceptor de Radiação Restrita"
palavras_chave_roteador = [
    'router', 'roteador', 'gateway', 'archer', 'echolife', 
    'optixstar', 'zxhn', 'hg8', 'ont', 'onu', 'h196', 'h1600', 'h3600'
]

# Prepara colunas auxiliares em minúsculas para busca de texto
df['Modelo_Lower'] = df['Modelo'].astype(str).str.lower()
df['Comercial_Lower'] = df['Nome Comercial'].astype(str).str.lower()

# Condição A: Pertence aos tipos oficiais de fibra e redes
filtro_tipo = df['Tipo do Produto'].isin(tipos_fibra_rede)

# Condição B: É transceptor de radiação restrita, mas possui nome/modelo de roteador ou ONT/ONU
filtro_transceptor_roteador = (
    (df['Tipo do Produto'] == 'Transceptor de Radiação Restrita') & 
    (
        df['Modelo_Lower'].str.contains('|'.join(palavras_chave_roteador), na=False) |
        df['Comercial_Lower'].str.contains('|'.join(palavras_chave_roteador), na=False)
    )
)

# 4. Aplica os filtros combinados
df_final = df[filtro_tipo | filtro_transceptor_roteador].copy()

# Remove colunas auxiliares e itens duplicados
df_final = df_final.drop(columns=['Modelo_Lower', 'Comercial_Lower'], errors='ignore')
df_final = df_final.drop_duplicates()

# Exibe o resumo do resultado
print(f"Total de equipamentos filtrados: {len(df_final)}")
print("\nDistribuição por Tipo de Produto:")
print(df_final['Tipo do Produto'].value_counts())

# 5. Salva o resultado final limpo em um novo arquivo CSV
df_final.to_csv('onts_onus_olts_roteadores_filtrados.csv', index=False, encoding='utf-8-sig')
print("\nArquivo salvo com sucesso como 'onts_onus_olts_roteadores_filtrados.csv'!")