import pandas as pd
import requests
import os
import time
import urllib.parse

# 1. Carrega a planilha com os links especializados por fabricante
input_file = 'equipamentos_com_links_especializados.csv'
output_file = 'relatorio_download_manuais.csv'

df = pd.read_csv(input_file, low_memory=False)

# Cria pasta local para armazenar os arquivos baixados
download_folder = 'manuais_pdf'
os.makedirs(download_folder, exist_ok=True)

if 'Status_Download' not in df.columns:
    df['Status_Download'] = ""
if 'Caminho_Arquivo_Local' not in df.columns:
    df['Caminho_Arquivo_Local'] = ""

# Headers simulando um navegador Chrome real
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
                  "AppleWebKit/537.36 (KHTML, like Gecko) " +
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/pdf,*/*;q=0.8"
}

print(f"Iniciando o processo de download para {len(df)} equipamentos...\n")

for index, row in df.iterrows():
    fabricante = str(row['Nome do Fabricante']).strip().replace('/', '_').replace('\\', '_')
    modelo = str(row['Modelo']).strip().replace('/', '_').replace('\\', '_')
    url = str(row['Link_Especializado']).strip()
    
    # Nome padronizado para o arquivo local
    file_name = f"{fabricante}_{modelo}.pdf".replace(" ", "_")
    file_path = os.path.join(download_folder, file_name)
    
    # Se o PDF válido já existe no disco, ignora para poupar tempo (checkpoint)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 5000:
        df.at[index, 'Status_Download'] = "Baixado anteriormente (Cache)"
        df.at[index, 'Caminho_Arquivo_Local'] = file_path
        continue

    try:
        response = requests.get(url, headers=headers, timeout=12, stream=True)
        
        if response.status_code == 200:
            # Salva temporariamente para validação
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validação: Checa se o arquivo tem tamanho > 5KB e cabeçalho de PDF (%PDF)
            is_valid_pdf = False
            if os.path.getsize(file_path) > 5000:
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header.startswith(b'%PDF'):
                        is_valid_pdf = True
            
            if is_valid_pdf:
                df.at[index, 'Status_Download'] = "Sucesso (PDF Válido)"
                df.at[index, 'Caminho_Arquivo_Local'] = file_path
                print(f"[{index+1}/{len(df)}] PDF Baixado: {file_name}")
            else:
                # Se baixou uma página HTML de busca/erro em vez de um PDF direto
                os.remove(file_path)
                df.at[index, 'Status_Download'] = "Link requer navegação/download manual"
                print(f"[{index+1}/{len(df)}] Paginada de busca/HTML retornada para: {modelo}")
        else:
            df.at[index, 'Status_Download'] = f"Erro HTTP {response.status_code}"
            print(f"[{index+1}/{len(df)}] HTTP {response.status_code} em {modelo}")
            
    except Exception as e:
        df.at[index, 'Status_Download'] = f"Falha de conexão: {str(e)}"
        print(f"[{index+1}/{len(df)}] Exceção de rede em {modelo}")
        
    # Salva o relatório atualizado a cada item (Checkpointing)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    time.sleep(1.0)

print(f"\nProcesso concluído! Relatório salvo em '{output_file}'.")