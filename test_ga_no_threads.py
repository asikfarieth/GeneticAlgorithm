import unittest
import population
import simulation
import genome
import creature
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=10, gene_count=3)
        sim = simulation.Simulation()

        avg_fitnesses = []
        avg_links_list = []

        for iteration in range(1000):
            for cr in pop.creatures:
                sim.run_creature(cr, 2400)
            fits = [cr.get_distance_travelled() for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]

            avg_fitness = np.mean(fits)
            avg_links = np.mean(links)

            avg_fitnesses.append(avg_fitness)
            avg_links_list.append(avg_links)

            if iteration % 20 == 0:
                print("Generation:", iteration, "Average fitness:", np.round(avg_fitness, 3),
                      "Mean links", np.round(avg_links), "Max links", np.round(np.max(links)))

            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)

            if iteration % 10 == 0:
                max_fit = np.max(fits)
                for cr in pop.creatures:
                    if cr.get_distance_travelled() == max_fit:
                        new_cr = creature.Creature(1)
                        new_cr.update_dna(cr.dna)
                        new_creatures[0] = new_cr
                        filename = "elite_"+str(iteration)+".csv"
                        genome.Genome.to_csv(cr.dna, filename)
                        break

            pop.creatures = new_creatures

        # Create a CSV file
        with open('data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Average Fitness", "Average Links"])
            for i in range(0, 1000, 250):
                writer.writerow([i, avg_fitnesses[i], avg_links_list[i]])

        # Plotting the average fitness
        plt.plot(avg_fitnesses)
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.show()

        # Creating a table with pandas
        df = pd.DataFrame({'Average Fitness': avg_fitnesses, 'Average Links': avg_links_list})
        print(df)

        self.assertNotEqual(fits[0], 0)

unittest.main()
