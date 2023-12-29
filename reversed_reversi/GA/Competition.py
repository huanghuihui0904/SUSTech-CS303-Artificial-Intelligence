from ReversiSimulator import *
import ai_main
import random


def create_population(size):
    population_list = []

    for i in range(size):
        mobility_weight = (random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10)
        frontier_weight = (random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10)
        stability_weight = (random.random() * 10, random.random() * 10)
        position_weight = (random.random() * 10, random.random() * 10)
        chess_counter_weight = (random.random() * 10, random.random() * 10, random.random() * 10, random.random() * 10)
        ai = ai_main.AI(8, -1, 5)
        ai.set_weight(mobility_weight, frontier_weight, stability_weight, position_weight, chess_counter_weight)
        population_list.append(ai)
    return population_list


def population_variation(good_count, populaiton_list):
    good_populaiton_list = populaiton_list[:good_count]
    variation_population_list=[]
    #add NO1
    variation_population_list.append(populaiton_list[0])
    #variate NO1
    new_mobility_weight = (good_populaiton_list[0].mobility_weight[0] + random.random(),
                           good_populaiton_list[0].mobility_weight[1] + random.random(),
                           good_populaiton_list[0].mobility_weight[2] + random.random(),
                           good_populaiton_list[0].mobility_weight[3] + random.random())
    new_frontier_weight = (good_populaiton_list[0].frontier_weight[0] + random.random(),
                           good_populaiton_list[0].frontier_weight[1] + random.random(),
                           good_populaiton_list[0].frontier_weight[2] + random.random(),
                           good_populaiton_list[0].frontier_weight[3] + random.random())
    new_stability_weight = (good_populaiton_list[0].stability_weight[0] + random.random(),
                            good_populaiton_list[0].stability_weight[1] + random.random())
    new_position_value_weight = (good_populaiton_list[0].position_value_weight[0] + random.random(),
                                 good_populaiton_list[0].position_value_weight[1] + random.random())
    new_chess_counter_weight = (good_populaiton_list[0].chess_counter_weight[0] + random.random(),
                                good_populaiton_list[0].chess_counter_weight[1] + random.random(),
                                good_populaiton_list[0].chess_counter_weight[2] + random.random(),
                                good_populaiton_list[0].chess_counter_weight[3] + random.random())
    ai = ai_main.AI(8, -1, 5)
    ai.set_weight(new_mobility_weight, new_frontier_weight, new_stability_weight,
                            new_position_value_weight, new_chess_counter_weight)
    variation_population_list.append(ai)

    #variate others
    for i in range(1, good_count):

        new_mobility_weight = (good_populaiton_list[i].mobility_weight[0] + random.random(),
                               good_populaiton_list[i].mobility_weight[1] + random.random(),
                               good_populaiton_list[i].mobility_weight[2] + random.random(),
                               good_populaiton_list[i].mobility_weight[3] + random.random())
        new_frontier_weight = (good_populaiton_list[i].frontier_weight[0] + random.random(),
                               good_populaiton_list[i].frontier_weight[1] + random.random(),
                               good_populaiton_list[i].frontier_weight[2] + random.random(),
                               good_populaiton_list[i].frontier_weight[3] + random.random())
        new_stability_weight = (good_populaiton_list[i].stability_weight[0] + random.random(),
                                good_populaiton_list[i].stability_weight[1] + random.random())
        new_position_value_weight = (good_populaiton_list[i].position_value_weight[0] + random.random(),
                                     good_populaiton_list[i].position_value_weight[1] + random.random())
        new_chess_counter_weight = (good_populaiton_list[i].chess_counter_weight[0] + random.random(),
                                    good_populaiton_list[i].chess_counter_weight[1] + random.random(),
                                    good_populaiton_list[i].chess_counter_weight[2] + random.random(),
                                    good_populaiton_list[i].chess_counter_weight[3] + random.random())
        ai = ai_main.AI(8, -1, 5)
        ai.set_weight(new_mobility_weight, new_frontier_weight, new_stability_weight,
                                           new_position_value_weight, new_chess_counter_weight)
        variation_population_list.append(ai)

    for i in range(1, good_count):
        new_mobility_weight = (good_populaiton_list[i].mobility_weight[0] - random.random(),
                               good_populaiton_list[i].mobility_weight[1] - random.random(),
                               good_populaiton_list[i].mobility_weight[2] - random.random(),
                               good_populaiton_list[i].mobility_weight[3] - random.random())
        new_frontier_weight = (good_populaiton_list[i].frontier_weight[0] - random.random(),
                               good_populaiton_list[i].frontier_weight[1] - random.random(),
                               good_populaiton_list[i].frontier_weight[2] - random.random(),
                               good_populaiton_list[i].frontier_weight[3] - random.random())
        new_stability_weight = (good_populaiton_list[i].stability_weight[0] - random.random(),
                                good_populaiton_list[i].stability_weight[1] - random.random())
        new_position_value_weight = (good_populaiton_list[i].position_value_weight[0] - random.random(),
                                     good_populaiton_list[i].position_value_weight[1] - random.random())
        new_chess_counter_weight = (good_populaiton_list[i].chess_counter_weight[0] - random.random(),
                                    good_populaiton_list[i].chess_counter_weight[1] - random.random(),
                                    good_populaiton_list[i].chess_counter_weight[2] - random.random(),
                                    good_populaiton_list[i].chess_counter_weight[3] - random.random())
        good_populaiton_list[i].set_weight(new_mobility_weight, new_frontier_weight, new_stability_weight,
                                           new_position_value_weight, new_chess_counter_weight)
        ai = ai_main.AI(8, -1, 5)
        ai.set_weight(new_mobility_weight, new_frontier_weight, new_stability_weight,
                                new_position_value_weight, new_chess_counter_weight)
        variation_population_list.append(ai)


    return variation_population_list


def variation_first(variate_count,good_ai):
    variation_population_list=[]
    #variate
    for i in range(variate_count):

        new_mobility_weight = (good_ai.mobility_weight[0] + random.random(),
                               good_ai.mobility_weight[1] + random.random(),
                               good_ai.mobility_weight[2] + random.random(),
                               good_ai.mobility_weight[3] + random.random())
        new_frontier_weight = (good_ai.frontier_weight[0] + random.random(),
                               good_ai.frontier_weight[1] + random.random(),
                               good_ai.frontier_weight[2] + random.random(),
                               good_ai.frontier_weight[3] + random.random())
        new_stability_weight = (good_ai.stability_weight[0] + random.random(),
                                good_ai.stability_weight[1] + random.random())
        new_position_value_weight = (good_ai.position_value_weight[0] + random.random(),
                                     good_ai.position_value_weight[1] + random.random())
        new_chess_counter_weight = (good_ai.chess_counter_weight[0] + random.random(),
                                    good_ai.chess_counter_weight[1] + random.random(),
                                    good_ai.chess_counter_weight[2] + random.random(),
                                    good_ai.chess_counter_weight[3] + random.random())
        ai = ai_main.AI(8, -1, 5)
        ai.set_weight(new_mobility_weight, new_frontier_weight, new_stability_weight,
                                           new_position_value_weight, new_chess_counter_weight)
        variation_population_list.append(ai)

    for i in range(variate_count):
        new_mobility_weight = (good_ai.mobility_weight[0] - random.random(),
                               good_ai.mobility_weight[1] - random.random(),
                               good_ai.mobility_weight[2] - random.random(),
                               good_ai.mobility_weight[3] - random.random())
        new_frontier_weight = (good_ai.frontier_weight[0] - random.random(),
                               good_ai.frontier_weight[1] - random.random(),
                               good_ai.frontier_weight[2] - random.random(),
                               good_ai.frontier_weight[3] - random.random())
        new_stability_weight = (good_ai.stability_weight[0] - random.random(),
                                good_ai.stability_weight[1] - random.random())
        new_position_value_weight = (good_ai.position_value_weight[0] - random.random(),
                                     good_ai.position_value_weight[1] - random.random())
        new_chess_counter_weight = (good_ai.chess_counter_weight[0] - random.random(),
                                    good_ai.chess_counter_weight[1] - random.random(),
                                    good_ai.chess_counter_weight[2] - random.random(),
                                    good_ai.chess_counter_weight[3] - random.random())

        ai = ai_main.AI(8, -1, 5)
        ai.set_weight(new_mobility_weight, new_frontier_weight, new_stability_weight,
                                new_position_value_weight, new_chess_counter_weight)
        variation_population_list.append(ai)


    return variation_population_list


NUM=3
ITERATION=2


# population_list = create_population(NUM)

ai_first = ai_main.AI(8, -1, 5)

mobility_weight = (7.431984110486266, 19.53297002767078, 23.48944484467045, 13.381247157730046)
frontier_weight = (27.40059735771842, 19.81351374920519, 25.39227535409172, 23.688432516469238)
stability_weight =   (18.745328833713923, 28.65157026667172)
position_value_weight =  (33.6218396954897, 12.357885942821804)
chess_counter_weight = (19.625177529845352, 20.00164854166681, 15.186890993244605, 17.273328454158936)
ai_first.set_weight(mobility_weight, frontier_weight, stability_weight,
              position_value_weight, chess_counter_weight)

population_list = variation_first(NUM,ai_first)

for i in range(ITERATION):
    score_ai_result = evaluate_competition([0, population_list])
    best=score_ai_result[0][1]
    print("self.mobility_weight= ",best.mobility_weight)
    print("self.frontier_weight= ",best.frontier_weight)
    print("self.stability_weight= ",best.stability_weight)
    print("self.position_value_weight= ",best.position_value_weight)
    print("self.chess_counter_weight= ",best.chess_counter_weight)
    print("------------------------------------------------------------------------------------------------------------------------------")
    # new_population_list = [item[1] for item in score_ai_result]
    # variation_population_list = population_variation(int(NUM/2), new_population_list)
    # population_list=variation_population_list
    population_list=variation_first(NUM,best)





