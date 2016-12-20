#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import subprocess
import os
import random
import math
import qgrid

# Machine Learning imports
import cma
from sklearn import tree

from econometrics import gini
from econometrics import draw_ginis
import pandas as pd


class RepresentativityNotDefinedException(Exception):
    print ('Please provide the number of individuals and their representativity in the population')
    pass


class Reform():
    """Reform is a class that create a reform optimized on multiple variable.

        It takes a population, a target variable, and parameters for the optimization and returns a reform.

    """

    def __init__(self,
                 population,
                 target_variable,
                 benefits_constraints=[],
                 taxes_constraints=[],
                 tax_threshold_constraints=[],
                 taxable_variable=None,
                 max_cost=0,
                 min_saving=0,
                 verbose=False,
                 percent_not_pissed_off=0,
                 weights=None,
                 price_of_no_regression=0,
                 max_evals = 5000):
        """
            Args:
                population: The class that handles the population.
                target_variable: The variable we try to predict , usually the available income.
                taxable_variable: The variable on which we want to take a tax, if needed.
        """

        self._target_variable = target_variable
        self._taxable_variable = taxable_variable
        self._max_cost = 0
        self._price_of_no_regression = price_of_no_regression
        self._population = population
        self._percent_not_pissed_off = percent_not_pissed_off
        self._max_cost = max_cost
        self._min_saving = min_saving
        self._weights = weights

        if (self._max_cost != 0 or self._min_saving != 0) and self._population._representativity is None:
            raise RepresentativityNotDefinedException()

        if verbose:
            cma.CMAOptions('verb')

        self.init_parameters(benefits_constraints=benefits_constraints,
                             taxes_constraints=taxes_constraints,
                             tax_threshold_constraints=tax_threshold_constraints)

        # new_parameters = self.add_segments(direct_parameters,  barem_parameters)
        # self.init_parameters(new_parameters)

        res = cma.fmin(self.objective_function, self._all_coefs, 10000.0, options={'maxfevals': max_evals})

        # print '\n\n\n Reform proposed: \n'
        #
        final_parameters = []

        i = 0
        while i < len(self._index_to_variable):
            final_parameters.append({'variable': self._index_to_variable[i],
                                     'value': res[0][i],
                                     'type': 'benefit'})
            i += 1

        offset = len(self._index_to_variable)

        while i < offset + len(self._tax_rate_parameters):
            final_parameters.append({'variable': self._tax_rate_parameters[i-offset],
                                     'value': res[0][i],
                                     'type': 'tax_rate'})
            i += 1

        offset = len(self._index_to_variable) + len(self._tax_rate_parameters)
        while i < offset + len(self._tax_threshold_parameters):
            final_parameters.append({'variable': self._tax_threshold_parameters[i-offset],
                                     'value': res[0][i],
                                     'type': 'tax_threshold'})
            i += 1

        self.final_parameters = final_parameters
        self.simulated_results, self.error, self.cost, self.pissed = self.apply_reform_on_population(self._population._population, coefficients=res[0])


    def is_optimized_variable(self, var):
        return var != self._taxable_variable and var != self._target_variable

    def init_parameters(self, benefits_constraints, taxes_constraints=[], tax_threshold_constraints=[]):
        print repr(benefits_constraints)
        var_total = {}
        var_occurences = {}
        self._index_to_variable = []
        self._all_coefs = []
        self._var_to_index = {}
        self._var_tax_rate_to_index = {}
        self._var_tax_threshold_to_index = {}
        self._tax_rate_parameters = []
        self._tax_threshold_parameters = []
        self._parameters = set(benefits_constraints)
        index = 0
        for person in self._population._population:
            for var in benefits_constraints:
                if var in person:
                    if var not in self._var_to_index:
                        self._index_to_variable.append(var)
                        var_total[var] = 0
                        var_occurences[var] = 0
                        self._var_to_index[var] = index
                        index += 1
                    var_total[var] = var_total.get(var, 0) + person[var]
                    var_occurences[var] = var_occurences.get(var, 0) + 1

        for var in self._index_to_variable:
            self._all_coefs.append(var_total[var] / var_occurences[var])

        for var in taxes_constraints:
            self._all_coefs.append(0)
            self._var_tax_rate_to_index[var] = index
            self._tax_rate_parameters.append(var)
            index += 1

        for var in tax_threshold_constraints:
            self._all_coefs.append(5000)
            self._var_tax_threshold_to_index[var] = index
            self._tax_threshold_parameters.append(var)
            index += 1

    def find_all_possible_inputs(self, input_variable):
        possible_values = set()
        for person in self._population._population:
            if input_variable in person:
                if person[input_variable] not in possible_values:
                    possible_values.add(person[input_variable])
        return sorted(possible_values)

    def find_min_values(self, input_variable, output_variable):
        min_values = {}
        for person in self._population._population:
            if input_variable not in person:
                continue
            input = person[input_variable]
            if person[output_variable] <= min_values.get(input, 100000):
                min_values[input] = person[output_variable]
        return min_values

    def find_average_values(self, input_variable, output_variable):
        values = {}
        number_of_values = {}
        for person in self._population._population:
            if input_variable not in person:
                continue
            input = person[input_variable]
            values[input] = values.get(input, 0) + person[output_variable]
            number_of_values[input] = number_of_values.get(input, 0) + 1
        for input in values:
            values[input] = values[input] / number_of_values[input]
        return values

    def find_jumps_rec(self, init_jump_size, possible_inputs, values):
        if init_jump_size > 10000:
            return
        jumps = []
        for i in range(1, len(possible_inputs)):
            if abs(values[possible_inputs[i]] - values[possible_inputs[i-1]]) > init_jump_size:
                jumps.append(possible_inputs[i])

        if len(jumps) > 0 and len(jumps) < 5:
            return jumps
        else:
            return self.find_jumps_rec(init_jump_size * 1.1 , possible_inputs, values)

    def find_jumps(self, input_variable, output_variable, jumpsize=10, maxjumps=5, method='min'):
        """
            This function find jumps in the data

        """
        possible_inputs = self.find_all_possible_inputs(input_variable)

        # For binary values, jump detection is useless
        if len(possible_inputs) < 3:
            print 'No segmentation made on variable ' + input_variable + ' because it has less than 3 possible values'
            return []

        if method == 'min':
            values = self.find_min_values(input_variable, output_variable)
        elif method == 'average':
            values = self.find_average_values(input_variable, output_variable)
        else:
            assert False, 'Method to find the average value is badly defined, it should be "min" or "average"'

        jumps = self.find_jumps_rec(jumpsize, possible_inputs, values)

        if len(jumps) <= maxjumps:
            return jumps
        else:
            print 'No segmentation made on variable ' + input_variable + ' because it has more than ' \
                                                                       + str(maxjumps + 1) + ' segments'
            return []

    def add_segments_for_variable(self, variable):
        jumps = self.find_jumps(variable, self._target_variable)

        print 'Jumps for variable ' + variable + ' are ' + repr(jumps)

        if len(jumps) == 0:
            return []

        segment_names = []

        # First segment
        segment_name = variable + ' < ' + str(jumps[0])
        segment_names.append(segment_name)
        for person in self._population._population:
            if variable in person and person[variable] < jumps[0]:
                person[segment_name] = 1

        # middle segments
        for i in range(1, len(jumps)):
            if abs(jumps[i-1]-jumps[i]) > 1:
                segment_name = str(jumps[i-1]) + ' <= ' + variable + ' < ' + str(jumps[i])
            else:
                segment_name = variable + ' is ' + str(jumps[i-1])
            segment_names.append(segment_name)
            for person in self._population._population:
                if variable in person and person[variable] >= jumps[i-1] and person[variable] < jumps[i]:
                    person[segment_name] = 1

        # end segment
        segment_name = variable + ' >= ' + str(jumps[-1])
        segment_names.append(segment_name)
        for person in self._population._population:
            if variable in person and person[variable] >= jumps[-1]:
                person[segment_name] = 1

        return segment_names

    def add_segments(self, parameters, segmentation_parameters):
        new_parameters = []
        for variable in segmentation_parameters:
            new_parameters = new_parameters + self.add_segments_for_variable(variable)
        new_parameters = sorted(new_parameters)
        return parameters + new_parameters

    def simulated_target(self, person, coefs):
        simulated_target = 0
        threshold = 0
        tax_rate = 0
        for var in person:
            if var in self._parameters:
                idx = self._var_to_index[var]

                # Adding linear constant
                simulated_target += coefs[idx] * person[var]
            if var in self._tax_threshold_parameters:
                idx = self._var_tax_threshold_to_index[var]

                # determining the threshold from which we pay the tax
                threshold += coefs[idx] * person[var]
            if var in self._tax_rate_parameters:
                idx = self._var_tax_rate_to_index[var]

                # determining the tax_rate, divided by 100 to help the algorithm converge faster
                tax_rate += coefs[idx] * person[var] / 100

        simulated_target += person[self._taxable_variable] - (person[self._taxable_variable] - threshold / 10) * tax_rate
        return simulated_target

    def compute_cost_error(self, simulated, person):
        cost = simulated - person[self._target_variable]
        error = abs(cost)
        error_util = error / (person[self._target_variable] + 1)

        if cost < 0:
            pissed = 1
        else:
            pissed = 0
        return cost, error, error_util, pissed

    def objective_function(self, coefs):
        error = 0
        error2 = 0
        total_cost = 0
        pissed_off_people = 0

        nb_people = len(self._population._population)

        for person in self._population._population:
            simulated = self.simulated_target(person, coefs)
            this_cost, this_error, this_error_util, this_pissed = self.compute_cost_error(simulated, person)
            total_cost += this_cost
            error += this_error
            error2 += this_error_util * this_error_util
            pissed_off_people += this_pissed

        percentage_pissed_off = float(pissed_off_people) / float(nb_people)

        if random.random() > 0.98:
            print 'Best: avg change per month: ' + repr(int(error / (12 * len(self._population._population))))\
                  + ' cost: ' \
                  + repr(int(self.normalize_on_population(total_cost) / 1000000))\
                  + ' M/year and '\
                  + repr(int(1000 * percentage_pissed_off)/10) + '% people with lower salary'

        cost_of_overbudget = 100

        if self.normalize_on_population(total_cost) > self._max_cost:
            error2 += pow(cost_of_overbudget, 2) * self.normalize_on_population(total_cost)

        if -self.normalize_on_population(total_cost) < self._min_saving:
            error2 += pow(cost_of_overbudget, 2) * self.normalize_on_population(total_cost)

        return math.sqrt(error2)

    def find_useful_parameters(self, results, threshold=100):
        """
            Eliminate useless parameters
        """
        new_parameters = []
        optimal_values = []
        for i in range(len(results)):
            if results[i] >= threshold:
                new_parameters.append(self._index_to_variable[i])
                optimal_values.append(results[i])
            else:
                print 'Parameter ' + self._index_to_variable[i] + ' was dropped because it accounts to less than '\
                      + str(threshold) + ' euros'
        return new_parameters, optimal_values

    def population_to_input_vector(self, population):
        output = []
        for person in population:
            person_output = self.person_to_input_vector(person)
            output.append(person_output)
        return output

    def person_to_input_vector(self, person):
        return list(person.get(var, 0) for var in self._index_to_variable)

    def suggest_reform_tree(self,
                            parameters,
                            max_cost=0,
                            min_saving=0,
                            verbose=False,
                            max_depth=3,
                            image_file=None,
                            min_samples_leaf=2):
        self._max_cost = max_cost
        self._min_saving = min_saving

        if (self._max_cost != 0 or self._min_saving != 0) and self._population._representativity is None:
            raise RepresentativityNotDefinedException()

        self.init_parameters(parameters)

        X = self.population_to_input_vector(self._population._population)
        y = map(lambda x: int(x[self._target_variable]), self._population._population)

        clf = tree.DecisionTreeRegressor(max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf)
        clf = clf.fit(X, y)

        self.simulated_results, self.error, self.cost, self.pissed = \
            self.apply_reform_on_population._population(self._population._population, decision_tree=clf)

        if image_file is not None:
            with open( image_file + ".dot", 'w') as f:
                f = tree.export_graphviz(clf,
                                         out_file=f,
                                         feature_names=self._index_to_variable,
                                         filled=True,
                                         impurity=True,
                                         proportion=True,
                                         rounded=True,
                                         rotate=True
                                         )
            os.system('dot -Tpng ' + image_file + '.dot -o ' + image_file + '.png')
            # 'dot -Tpng enfants_age.dot -o enfants_age.png')
                             # ')

        # dot_data = tree.export_graphviz(clf)
        #
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # graph.write_pdf("new_law.pdf")
        #
        # dot_data = tree.export_graphviz(clf, out_file=None,
        #                          feature_names=self._index_to_variable,
        #                          filled=True, rounded=True,
        #                          special_characters=True)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # Image(graph.create_png())

    def is_boolean(self, variable):
        """
            Defines if a variable only has boolean values

        :param variable: The name of the variable of interest
        :return: True if all values are 0 or 1, False otherwise
        """
        for person in self._population._population:
            if variable in person and person[variable] not in [0, 1]:
                return False
        return True

    def apply_reform_on_population(self, population, coefficients=None, decision_tree=None):
        """
            Computes the reform for all the population

        :param population:
        :param coefficients:
        :return:
        """
        simulated_results = []
        total_error = 0
        total_cost = 0
        pissed = 0
        for i in range(0, len(population)):
            if decision_tree:
                simulated_result = float(decision_tree.predict(self.person_to_input_vector(population[i]))[0])
            elif coefficients is not None:
                simulated_result = self.simulated_target(population[i], coefficients)
            simulated_results.append(simulated_result)
            this_cost, this_error, this_error_util, this_pissed = self.compute_cost_error(simulated_result, population[i])
            total_cost += this_cost
            total_error += this_error
            pissed += this_pissed

        total_cost = self.normalize_on_population(total_cost)

        return simulated_results, total_error / len(population), total_cost, pissed / float(len(population))

    def normalize_on_population(self, cost):
        if self._population._representativity is None or self._population._representativity == 0:
            raise RepresentativityNotDefinedException()
        return cost / self._population._representativity

    def load_from_json(self, filename):
        with open('../data/' + filename, 'r') as f:
            return json.load(f)

    def show_stats(self):
        if self.cost > 0:
            display_cost_name = 'Total Cost in Euros'
            display_cost_value = int(self.cost)
        else:
            display_cost_name = 'Total Savings in Euros'
            display_cost_value = -int(self.cost)

        old_gini = gini(list(x[self._target_variable] for x in self._population._population))
        new_gini = gini(self.simulated_results)

        result_frame = pd.DataFrame({
            display_cost_name: [display_cost_value],
            'Average change / family / month in Euros' : [int(self.error) / 12],
            'People losing money' : str(100 * self.pissed) + '%',
            'Old Gini' : old_gini,
            'New Gini' : new_gini,
            })

        result_frame.set_index(display_cost_name, inplace=True)
        qgrid.show_grid(result_frame)

    def draw_ginis(self):
        draw_ginis(list(x[self._target_variable] for x in self._population._population), self.simulated_results)

    def show_constraints_value(self, constraint):
        coefficients = []
        variables = []
        for parameter in self.final_parameters:
            if parameter['type'] == constraint:
                coefficients.append(parameter['value'])
                variables.append(parameter['variable'])

        result_frame = pd.DataFrame({'Variables': variables, constraint + ' coef': coefficients})
        result_frame.set_index('Variables', inplace=True)
        qgrid.show_grid(result_frame)
