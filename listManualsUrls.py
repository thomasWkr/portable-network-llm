import pandas as pd
import urllib.parse

# 1. Carrega a base filtrada de ONTs, ONUs, OLTs e roteadores
input_file = 'onts_onus_olts_roteadores_filtrados.csv'
output_file = 'equipamentos_com_links_especializados.csv'

df = pd.read_csv(input_file, low_memory=False)

# 2. Função de roteamento de link inteligente baseada no fabricante
def generate_smart_link(row):
    fab = str(row['Nome do Fabricante']).lower()
    modelo = str(row['Modelo']).strip()
    modelo_encoded = urllib.parse.quote(modelo)
    
    if 'intelbras' in fab:
        return f"https://www.intelbras.com/pt-br/busca?s={modelo_encoded}"
    elif 'tp-link' in fab:
        return f"https://www.tp-link.com/br/search/?q={modelo_encoded}"
    elif 'huawei' in fab:
        return f"https://support.huawei.com/enterprise/en/search?keyword={modelo_encoded}"
    elif 'zte' in fab:
        return f"https://support.zte.com.cn/support/search/Search.aspx?key={modelo_encoded}"
    elif 'nokia' in fab:
        return f"https://www.nokia.com/networks/support/search?q={modelo_encoded}"
    else:
        # Fallback regulatório internacional para marcas menos comuns
        return f"https://fccid.io/search.php?q={modelo_encoded}"

# Aplica a regra para cada linha
df['Link_Especializado'] = df.apply(generate_smart_link, axis=1)

# 3. Salva a nova planilha estruturada
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"Planilha gerada com sucesso: '{output_file}' contendo URLs direcionadas por fabricante!")