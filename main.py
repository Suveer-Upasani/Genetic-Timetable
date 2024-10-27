import random
from deap import base, creator, tools, algorithms
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from prettytable import PrettyTable

# Constants
POPULATION_SIZE = 50
NUMB_OF_ELITE_SCHEDULES = 1
TOURNAMENT_SELECTION_SIZE = 1
MUTATION_RATE = 0.1
GENERATIONS = 2000

UNIVERSITY_START_TIME = datetime.strptime("08:30", "%H:%M")
UNIVERSITY_END_TIME = datetime.strptime("16:45", "%H:%M")
LUNCH_BREAK_START = datetime.strptime("12:45", "%H:%M")
LUNCH_BREAK_END = datetime.strptime("13:30", "%H:%M")
TIME_SLOT_DURATION = timedelta(hours=1)
LAB_TIME_SLOT_DURATION = timedelta(hours=2)
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
BREAKONE_START_TIME = datetime.strptime("10:30", "%H:%M")
BREAKONE_END_TIME = datetime.strptime("10:45", "%H:%M")
BREAKTWO_START_TIME = datetime.strptime("15:30", "%H:%M")
BREAKTWO_END_TIME = datetime.strptime("15:45", "%H:%M")

# Sample Data
subjects = ["DETT", "DS", "MMA", "OOCL", "PBL-I", "SCHIE", "ESD", "UE-I"]
panels = ["A", "B", "C", "D"]
professors = {
    "DETT": ["Dr. Ramaa Sandu", "Prof. Tejaswini Thaokar", "Dr. Vrushali Kulkarni"],
    "DS": ["Dr. Rashmi Phalnikar", "Dr. Vrushali Kulkarni", "Prof. Laxmi Bhagwat", "Dr. Pratvina Talele"],
    "MMA": ["Dr. Bharati Dixit", "Prof. Vidya Patil", "Prof. Seema Idhate", "Dr. Ranjana Agrawal"],
    "OOCL": ["Prof. Anita Gunjal", "Dr. Vitthal Gutte", "Prof. Rashmi Rane", "Dr. Amit Savyanavar"],
    "PBL-I": ["Prof. Vidya Patil", "Dr. Ranjana Agrawal"],
    "SCHIE": ["Prof. A"],
    "ESD": ["Prof. B"],
    "UE-I": ["Prof. C"]
}
time_slots = 8  # Total slots per day
days = len(DAYS_OF_WEEK)  # Total days per week

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_professor", lambda: random.choice([prof for prof_list in professors.values() for prof in prof_list]))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_professor, n=days * time_slots * len(panels) * len(subjects))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness Function with Enhanced Professor Check
def evaluate(individual):
    score = 0
    allocation_matrix = np.array(individual).reshape((days, time_slots, len(panels), len(subjects)))

    for day in range(days):
        for time in range(time_slots):
            current_time = UNIVERSITY_START_TIME + time * TIME_SLOT_DURATION

            # Skip over breaks
            if (LUNCH_BREAK_START <= current_time < LUNCH_BREAK_END or
                BREAKONE_START_TIME <= current_time < BREAKONE_END_TIME or
                BREAKTWO_START_TIME <= current_time < BREAKTWO_END_TIME):
                continue

            assigned_professors = set()  # Track professors already assigned in the time slot
            for panel in range(len(panels)):
                for subject_idx, subject in enumerate(subjects):
                    prof = allocation_matrix[day, time, panel, subject_idx]

                    # Check if professor is assigned multiple times in the same slot across panels
                    if prof in assigned_professors:
                        score += 1  # Penalize for double-booking
                    else:
                        assigned_professors.add(prof)  # Add professor to the set

                    # Check if professor's expertise matches the assigned subject
                    if prof not in professors[subject]:
                        score += 2  # Penalize if professor is teaching an incorrect subject

    return score,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUTATION_RATE)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SELECTION_SIZE)

# Running the Genetic Algorithm
def main():
    st.title("Professor Allotment Genetic Algorithm")

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(NUMB_OF_ELITE_SCHEDULES)

    # Genetic Algorithm Parameters
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=MUTATION_RATE, ngen=GENERATIONS, halloffame=hof, verbose=True)
    
    best_individual = hof[0]
    st.subheader("Best Schedule:")
    
    allocation_matrix = np.array(best_individual).reshape((days, time_slots, len(panels), len(subjects)))

    # Create a PrettyTable for the best schedule
    table = PrettyTable()
    table.field_names = ["Day/Time", "Panel A", "Panel B", "Panel C", "Panel D"]

    for day in range(days):
        for time in range(time_slots):
            current_time = UNIVERSITY_START_TIME + time * TIME_SLOT_DURATION
            if (LUNCH_BREAK_START <= current_time < LUNCH_BREAK_END or
                BREAKONE_START_TIME <= current_time < BREAKONE_END_TIME or
                BREAKTWO_START_TIME <= current_time < BREAKTWO_END_TIME):
                table.add_row([f"{DAYS_OF_WEEK[day]} ({current_time.strftime('%H:%M')})", "Break", "Break", "Break", "Break"])
                continue
            
            panel_assignments = []
            for panel in range(len(panels)):
                assignments = []
                for subject_idx, subject in enumerate(subjects):
                    prof = allocation_matrix[day, time, panel, subject_idx]
                    assignments.append(f"{subject} -> {prof}")
                panel_assignments.append(", ".join(assignments))
            table.add_row([f"{DAYS_OF_WEEK[day]} ({current_time.strftime('%H:%M')})"] + panel_assignments)
    
    st.write(table)

    fitness_value = evaluate(best_individual)
    st.write("Best Fitness Score:", fitness_value[0])

if __name__ == "__main__":
    main()
