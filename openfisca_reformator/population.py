# -*- coding: utf-8 -*-

import os
import json
import pkg_resources

import pandas as pd

package_dir = pkg_resources.get_distribution('OpenFisca-Reformator').location
data_dir = os.path.join(package_dir, 'data')

class EchantillonNotDefinedException(Exception):
    pass


class Population():
    """Population Manager is a class that loads a population from different sources: panda dataframe, json, openfisca.
       This population can be used for automatic inference of the law, or making parametric reforms.
       It contains methods for filtering and handling this population.
       It takes as input a population with data (e.g., salaries/taxes/subsidies) this population

       First a population is given in _raw_population
       Second, concepts (for instance, parent_isole) can be built and stored in _population
    """
    def __init__(self):
        pass

    @classmethod
    def from_csv(cls, base_name, nb_individuals, nb_individuals_ref):
        self = cls()

        population_df = pd.read_csv(os.path.join(data_dir, base_name))
        self._population = [
            row.to_dict()
            for index, row in population_df.iterrows()
            ]
        self._raw_population=[
            row.to_dict()
            for index, row in population_df.iterrows()
            ]

        self.compute_representativity(nb_individuals, nb_individuals_ref)

        return self

    @classmethod
    def from_openfisca_files(cls, base_name, variables_loaded_from_results, nb_individuals, nb_individuals_ref):
        def load_from_json(filename):
            with open(os.path.join(data_dir, filename), 'r') as f:
                return json.load(f)

        results_openfisca = load_from_json(base_name + '-openfisca.json')
        testcases = load_from_json(base_name + '-testcases.json')

        return cls.from_openfisca_json(results_openfisca, testcases, variables_loaded_from_results,
                                       nb_individuals, nb_individuals_ref)

    @classmethod
    def from_openfisca_json(cls, results, testcases, variables_loaded_from_results, nb_individuals, nb_individuals_ref):
        """Defining a new population from results computed with openfisca.
        """
        self = cls()

        self.compute_representativity(nb_individuals, nb_individuals_ref)
        self._raw_population = None
        self._population = []

        self._raw_population = testcases[:]
        self._population = [None] * len(testcases)
        for i  in range(len(self._population)):
            self._population[i] = {}

        for variable in variables_loaded_from_results:
            for i in range(len(results)):
                self._raw_population[i][variable] = results[i][variable]
                self._population[i][variable] = results[i][variable]

        return self

    def compute_representativity(self, nb_individuals, nb_individuals_ref):
        if nb_individuals_ref > 0:
            self._representativity = nb_individuals / nb_individuals_ref
        else:
            self._representativity = None

    def print_population_from_cerfa_values(self):
        total_people = 0
        for family in self._raw_population:
            total_people += 1
            if '0DB' in family and family['0DB'] == 1:
                total_people += 1
            if 'F' in family:
                total_people += family['F']
        print 'Total number of people in the population ' + repr(total_people)

    def compute_echantillion_from_cerfa_fields(self):
        total_people = 0
        for family in self._raw_population:
            total_people += 1
            if '0DB' in family and family['0DB'] == 1:
                total_people += 1
            if 'F' in family:
                total_people += family['F']

        # We assume that there are 2000000 people with RSA
        # TODO Put that where it belongs in the constructor
        # There are 62M inhabitant, 93% of which are represented on declaration of revenu
        self._representativity =  float(total_people) / (62000000 * 0.93)
        print 'Echantillon of ' + repr(total_people) + ' people, in percent of french population for similar revenu: ' + repr(100 * self._representativity) + '%'

    def declare_computed_variables(self, variables_to_functions):
        for var in variables_to_functions:
            self.declare_computed_variable(var, variables_to_functions[var])

    def declare_computed_variable(self, variable, function):
        for i in range(len(self._raw_population)):
            result = function(self._raw_population[i])
            if result is not None and result is not False:
                self._population[i][variable] = float(result)

    def filter_only_likely__population(self):
        """
            Removes unlikely elements in the population
            TODO: This should be done by the population generator

        :param: raw_population._population:
        :return: raw_population._population without unlikely cases
        """
        new_raw__population = []
        for case in self._raw_population._population:
            if (int(case['0DA']) <= 1950 or ('0DB' in case and int(case['0DA'] <= 1950))) and 'F' in case and int(case['F']) > 0:
                pass
            else:
                new_raw__population.append(case)
        self._raw_population = new_raw__population

    def filter_only_no_revenu(self):
        """
            Removes people who have a salary from the population

        :param raw_population._population:
        :return: raw_population._population without null salary
        """
        new_raw_population = []
        for case in self._raw_population._population:
            if case.get('1AJ', 0) < 1 and case.get('1BJ', 0) < 1:
                new_raw_population._population.append(case)
        self._raw_population._population = new_raw_population