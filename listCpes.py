import re

import pandas as pd


INPUT_FILE = 'input_sources/Produtos_Homologados_Anatel.csv'
OUTPUT_FILE = 'input_sources/onts_onus_olts_roteadores_filtrados.csv'
VALIDITY_COLUMN = 'Data de Validade do Certificado'

# Estas categorias identificam terminais ópticos, mas não distinguem GPON
# de outras tecnologias PON. A compatibilidade exige verificar cada modelo.
OPTICAL_TYPES = {
    'ONT - Terminação de Rede Ótica',
    'ONU - Unidade de Rede Óptica',
    'OLT - Terminação de Linha Óptica',
    'Terminal de Linha Óptica',
}
NETWORK_TYPES = {
    'Equipamento para Interconexão de Redes',
    'Equipamento de Rede de Dados',
    'Transceptor de Radiação Restrita',
}
# Limites de palavra evitam que "ont" selecione, por exemplo, "controle".
# Famílias comerciais não são prova suficiente de que o produto é roteador.
DEVICE_PATTERN = re.compile(r'\b(?:ont|onu|olt|router|roteador|roteadores)\b', re.I)


def filter_devices(df):
    """Seleciona candidatos ópticos/roteadores com validade posterior a 2020.

    Agrupa por fabricante e modelo, ignorando diferenças de nome comercial,
    maiúsculas/minúsculas e espaços. Mantém a maior validade de certificado.
    Registros sem fabricante/modelo só perdem duplicatas de linha inteira.
    Datas ausentes ou inválidas são excluídas.
    """
    descriptions = (
        df['Modelo'].fillna('') + ' ' + df['Nome Comercial'].fillna('')
    )
    device_mask = df['Tipo do Produto'].isin(OPTICAL_TYPES) | (
        df['Tipo do Produto'].isin(NETWORK_TYPES)
        & descriptions.str.contains(DEVICE_PATTERN, na=False)
    )
    validity = pd.to_datetime(df[VALIDITY_COLUMN], errors='coerce')
    candidates = df.loc[device_mask & validity.dt.year.gt(2020)].drop_duplicates().copy()
    # Chaves auxiliares ficam fora da saída para preservar as colunas originais.
    keys = candidates[['Nome do Fabricante', 'Modelo']].apply(
        lambda column: column.fillna('').str.normalize('NFKC')
        .str.strip().str.replace(r'\s+', ' ', regex=True).str.casefold()
    )
    identified = keys.ne('').all(axis=1)
    order = pd.DataFrame({
        'maker': keys['Nome do Fabricante'].to_numpy(),
        'model': keys['Modelo'].to_numpy(),
        'validity': pd.to_datetime(candidates[VALIDITY_COLUMN], errors='coerce').to_numpy(),
        'identified': identified.to_numpy(),
        'position': range(len(candidates)),
    })
    # Em empate, preserva a primeira ocorrência na entrada.
    keep = order.loc[order['identified']].sort_values(
        'validity', ascending=False, kind='stable'
    ).drop_duplicates(['maker', 'model'])['position']
    positions = sorted([*keep, *order.loc[~order['identified'], 'position']])
    return candidates.iloc[positions].copy()


def main():
    # Preserva zeros à esquerda nos números de homologação e CNPJs.
    df = pd.read_csv(INPUT_FILE, sep=';', dtype=str)
    result = filter_devices(df)
    print(f'Total de equipamentos candidatos filtrados: {len(result)}')
    print('\nDistribuição por Tipo de Produto:')
    print(result['Tipo do Produto'].value_counts())
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f'\nArquivo salvo com sucesso como {OUTPUT_FILE!r}!')
    print('A compatibilidade GPON deve ser confirmada na documentação de cada modelo.')


if __name__ == '__main__':
    main()
