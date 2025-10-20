# Sistema de Alocação de Horários Escolares

Sistema inteligente para geração automática de horários escolares utilizando Regressão Linear para previsão de demanda e Algoritmo Genético para otimização da alocação.
Descrição
Este sistema automatiza o processo de criação de horários escolares em três etapas principais:

Regressão Linear: Prevê o número de alunos por série com base em dados históricos
Geração de Turmas: Cria automaticamente as turmas baseadas nas previsões
Algoritmo Genético: Otimiza a alocação de aulas considerando restrições de salas, professores e horários

# Funcionalidades

Previsão automática de demanda de matrículas
Geração inteligente de turmas
Alocação otimizada de horários
Respeito a restrições de:

Capacidade das salas
Disponibilidade de professores
Turnos (manhã, tarde, noite)
Carga horária máxima


Relatórios detalhados de conflitos e otimizações
Exportação de resultados em CSV

Requisitos

Python 3.8 ou superior
pandas
numpy
scikit-learn
DEAP (Distributed Evolutionary Algorithms in Python)

Instalação
bash# Clone o repositório
git clone <seu-repositorio>
cd TCC_2

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install pandas numpy scikit-learn deap
Estrutura de Arquivos
TCC_2/
│
├── main.py                          # Script principal
├── linear_regression.py             # Módulo de regressão linear
├── genetic_algorithm.py             # Módulo do algoritmo genético
│
├── data/
│   ├── historical_enrollment_data.csv    # Dados históricos (OBRIGATÓRIO)
│   ├── rooms_data.csv                    # Dados das salas (OBRIGATÓRIO)
│   ├── teachers_data.csv                 # Dados dos professores (OBRIGATÓRIO)
│   ├── schedules_data.csv                # Horários disponíveis (OBRIGATÓRIO)
│   ├── teacher_assignments.csv           # Atribuições específicas (OPCIONAL)
│   └── classes_data.csv                  # Gerado automaticamente pelo STEP 0
│
└── README.md
Formato dos Arquivos de Entrada
1. historical_enrollment_data.csv (OBRIGATÓRIO)
Dados históricos de matrículas por série e ano.
csvyear,grade,num_students
2020,1,250
2020,2,230
2020,3,220
2021,1,260
2021,2,245
2021,3,225
2022,1,270
2022,2,255
2022,3,240
2023,1,280
2023,2,265
2023,3,250
Colunas:

year: Ano letivo
grade: Série/ano escolar (1, 2, 3, etc.)
num_students: Número de alunos matriculados

2. rooms_data.csv (OBRIGATÓRIO)
Informações sobre as salas disponíveis.
csvroom_id,capacity,type
SALA_01,35,Regular
SALA_02,35,Regular
SALA_03,40,Regular
LAB_01,30,Laboratório
LAB_02,30,Laboratório
Colunas:

room_id: Identificador único da sala
capacity: Capacidade máxima de alunos
type: Tipo da sala (Regular, Laboratório, etc.)

3. teachers_data.csv (OBRIGATÓRIO)
Informações sobre os professores.
csvteacher_id,name,main_subject,available_morning,available_afternoon,available_evening,max_workload
PROF_01,João Silva,Matemática,1,1,0,40
PROF_02,Maria Santos,Português,1,1,0,40
PROF_03,Carlos Souza,Ciências,1,0,0,30
PROF_04,Ana Costa,História,0,1,1,35
Colunas:

teacher_id: Identificador único do professor
name: Nome do professor
main_subject: Disciplina principal
available_morning: Disponível no turno da manhã (1 = sim, 0 = não)
available_afternoon: Disponível no turno da tarde (1 = sim, 0 = não)
available_evening: Disponível no turno da noite (1 = sim, 0 = não)
max_workload: Carga horária máxima semanal

4. schedules_data.csv (OBRIGATÓRIO)
Horários disponíveis para alocação.
csvschedule_id,weekday,shift,period
SEG_M_1,Segunda-feira,Manhã,1
SEG_M_2,Segunda-feira,Manhã,2
SEG_M_3,Segunda-feira,Manhã,3
SEG_M_4,Segunda-feira,Manhã,4
SEG_T_1,Segunda-feira,Tarde,1
SEG_T_2,Segunda-feira,Tarde,2
TER_M_1,Terça-feira,Manhã,1
TER_M_2,Terça-feira,Manhã,2
Colunas:

schedule_id: Identificador único do horário
weekday: Dia da semana
shift: Turno (Manhã, Tarde, Noite)
period: Período/aula dentro do turno

5. teacher_assignments.csv (OPCIONAL)
Atribuições fixas de professores a turmas específicas.
csvclass_id,teacher_id
1A,PROF_01
1B,PROF_01
2A,PROF_02
Colunas:

class_id: Identificador da turma
teacher_id: Identificador do professor

Uso
Execução Completa
bashpython main.py
O sistema executará automaticamente os três passos:
STEP 0: Regressão Linear

Carrega dados históricos
Treina modelos de previsão por série
Gera previsões para o próximo ano
Cria o arquivo classes_data.csv

STEP 1: Carregamento de Dados

Carrega todos os arquivos de entrada
Valida a consistência dos dados
Prepara estruturas para o algoritmo genético

STEP 2: Algoritmo Genético

Gera população inicial de soluções
Executa evolução por 80 gerações
Aplica operadores de crossover e mutação
Seleciona a melhor solução

STEP 3: Análise e Relatório

Gera relatório detalhado de conflitos
Exporta horários em formato legível
Salva estatísticas de otimização

Regenerar classes_data.csv
Se o arquivo classes_data.csv já existir, o sistema perguntará se deseja regenerá-lo:
Arquivo data/classes_data.csv já existe.
Deseja regerar o arquivo? (s/n):
Digite s para regenerar ou n para usar o arquivo existente.
Parâmetros do Algoritmo Genético
Os parâmetros podem ser ajustados na chamada de execute_genAlgorithm() em main.py:

ngen: Número de gerações (padrão: 80)
npop: Tamanho da população (padrão: 150)
Taxa de crossover: 0.7
Taxa de mutação: 0.2

Saídas Geradas
1. classes_data.csv
Turmas geradas automaticamente pela regressão linear.
2. Console Output
Relatório detalhado incluindo:

Fitness da melhor solução
Número de conflitos por categoria
Estatísticas de utilização de salas
Distribuição de carga horária dos professores

3. Logs de Execução
Informações sobre cada etapa do processamento.
Restrições e Penalizações
O algoritmo genético considera as seguintes restrições (com peso de penalização):

Conflitos de Horário (peso: 100)

Professor não pode dar duas aulas simultâneas
Turma não pode ter duas aulas no mesmo horário
Sala não pode ser usada por duas turmas simultaneamente


Capacidade da Sala (peso: 50)

Número de alunos não pode exceder a capacidade da sala
