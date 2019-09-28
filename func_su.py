############################## Part 2: All My Functions  ######################################

###### sample binary string b consistent with scenario x
def gen_b(x):
## input: x is a scenario, the outcome vector generated is consistent with x
    b = ['0'] * n;  
    for t in range(n): # generate an outcome vector consistent with x
        if input_table[x][str(t)] != 'u': # if not star 
            b[t] = input_table[x][str(t)]
        else: # if star, toss a coin
            b[t] = str(np.random.randint(0, 2) )
    return b

###### This func prints the alive scenarios
def print_alive_sce(alive_dict):
## Input: a dictionary
## keep track of a dict whose keys are the scenarios, with 0/1 value, indicating whether a sce- is alive
    print("alive scenarios: ")
    for sce in alive_dict:
        if alive_dict[sce] == 1:
            print(sce)
            
###### Compute the greedy score for a test (for the clean version)
def estimate_G(test, E, num_sample):
# input: 
    # test: an integer, the index of the test being considered
    # E: the set of tests already selected
    # num_sample: recall that we don't know how to compute the score efficiently, so we apply Monte Carlo for num_sample times.
    gain = np.zeros(num_sample) # store the gain
    partial_sum = 0

    for x in range(m): # rewrite the score in to 2 sums: first sum over x, then over b
        for k in range(num_sample): # perform Monte Carlo num_sample times
            b = gen_b(x) ## sample a binary string b
            
            ## compute how many scenarios already ruled out
            alive_dict = {i: 1 for i in range(m)} # 1 for alive, 0 for dead
                      
            for sce in alive_dict:
                for T in E: # check if sce is inconsistent with b_E 
                    ### may be costly when E large! (since #iter = |E|) ###
                    temp = input_table[sce][str(T)]  
                    if temp != 'u' and temp != b[T] and alive_dict[sce] == 1: # inconsistent label
                        alive_dict[sce] = 0
            
            num_alive = sum(alive_dict.values())
            
            ## compute greedy score:
            if num_alive == 1: # if only x_hat is alive
                gain[k] = 0
            elif num_alive == 0:
                print("Something went wrong! Num of alive becomes 0.")
            else: # compute marginal increment of test
                marginal_gain = 0 # extra ``kills'' of T
                for sce in alive_dict: # check if T can rule out x (under b)
                    temp = input_table[sce][str(test)] # look at entry (sce, test) 
                    if temp != 'u' and temp != b[test] and alive_dict[sce] == 1: # if indeed inconsitent
                        marginal_gain += 1 
                gain[k] = marginal_gain / num_alive

        partial_sum += q[x] * (np.sum(gain) / num_sample)

    return partial_sum


###### for equiv- class identification, we need a new function to estimate the score.
def estimate_G_new(new_test, E, num_sample):
# E = ordered subset of currently chosen tests
    sample_score = np.zeros(num_sample) # store the gain
    inner_product = 0 # final score

    for x in range(m): # rewrite the score as 2 sums: first sum over x, then over b
        #print("processing scenario", x)
        #print("number of stars containing it:", len(ECG[x]))
        for k in range(num_sample): # perform MonteCarlo num_sample times
            b = gen_b(x) ## sample from b from M_x  
            product_before = 1
            product_after = 1
            for y in ECG[x]: ## loop over all neibors of x (including x)
                ## compute how many scenarios already ruled out
                temp = list(range(m) )
                for z in ECG[y]: # we focus only on sce- outside Star(y)
                    temp.remove(z)
                num_sce_outside_star = len(temp)
                alive_dict_y = {i: 1 for i in temp}

                ## find num of alive sce- outside Star(y) before performing new_test
                for sce in alive_dict_y: 
                    for T in E: # check if sce is inconsistent with b_E 
                        ### may be costly when E large! (since #iter = |E|) ###
                        temp = input_table[sce][str(T)]  
                        if temp != 'u' and temp != b[T] and alive_dict_y[sce] == 1: # inconsistent label
                            alive_dict_y[sce] = 0

                num_alive_before = sum(alive_dict_y.values() )
                frac_alive_before = num_alive_before/num_sce_outside_star # 1- g_{S,b}(E)
                product_before *= frac_alive_before # need product over all stars inci- to x
                
                ##  same as above, after performing new_test
                for sce in alive_dict_y: 
                    temp = input_table[sce][str(new_test)]  
                    if temp != 'u' and temp != b[new_test] and alive_dict_y[sce] == 1: # inconsistent label
                        alive_dict_y[sce] = 0

                num_alive_after = sum(alive_dict_y.values() )
                frac_alive_after = num_alive_after/num_sce_outside_star # 1- g_{S,b}(E)
                product_after *= frac_alive_after # need product over all stars inci- to x

            if product_before != 0:
                sample_score[k] = (product_before - product_after)/ product_before # the score of sample k     
            else:
                sample_score[k] = 0
            #print("score of sample", k, ":", sample_score[k])         


        inner_product += q[x] * (np.sum(sample_score) / num_sample) # final score. Sum over all x 
    return inner_product


###### compute the cover time of a permutation 
def compute_cover_time(perm, outcome, skip_stupid_test): 
# input: 
    # perm = a permutation of tests
    # outcome = a realized outcome vector, each entry 1/0
    # skip_stupid_test: 0/1, whether or not skip a test which rules out nothing (``stupid test'')
    cover_time = 0
    alive_dict = {i: 1 for i in range(m)} # keep track of the status of sce, 1 for alive, 0 for dead
    
    if skip_stupid_test == 0:
        for t in range(len(E)):
            test_index = perm[t]
            num_alive = sum(alive_dict.values())
            if num_alive <= 1: # if there are at most 1 alive, declare done
                cover_time = t + 1
                break
            else: # at least two sce alive
                for x in alive_dict: # if x is inconsistent with outcome, then kill
                    if alive_dict[x] == 1 and outcome[test_index] != input_table[x][str(test_index)] and input_table[x][str(test_index)] != 'u': 
                        alive_dict[x] = 0 # x is ruled out

    elif skip_stupid_test == 1: 
        for t in range(len(E)):
            test_index = perm[t]
            num_alive = sum(alive_dict.values())
            if num_alive <= 1: # if there are at most 1 alive, declare done
                break
            else: # at least two sce alive
                marginal_gain = 0;
                for x in alive_dict: # if x is inconsistent with outcome, then kill
                    if alive_dict[x] == 1 and outcome[test_index] != input_table[x][str(test_index)] and input_table[x][str(test_index)] != 'u': 
                        alive_dict[x] = 0 # x is ruled out
                        marginal_gain += 1
                if marginal_gain > 0:
                    cover_time += 1
    else: 
        print("error!")
    return cover_time


###### compute the cover time of a permutation
## equiva- class idetification, with ''clique stopping criterion''
def compute_cover_time_ec_clique(perm, outcome, skip_stupid_test, ECG): 
# input: 
# perm = a permutation of tests
# outcome = a realized outcome vector, each entry 1/0
# skip_stupid_test: 0/1, whether or not skip a test which rules out nothing (``stupid test'')
    cover_time = 0
    alive_dict = {i: 1 for i in range(m)} # keep track of the status of sce, 1 for alive, 0 for dead
    stop_flag_1 = 0
    stop_flag_2 = 0

    if skip_stupid_test == 0:
        for t in range(len(perm)):
            test_index = perm[t]
            for x in alive_dict: # update alive_dict
                if alive_dict[x] == 1 and outcome[test_index] != input_table[x][str(test_index)] and input_table[x][str(test_index)] != 'u': 
                    alive_dict[x] = 0 # rule out x

            # check whether there is a star S s.t. \bar S is covered
            num_alive = sum(alive_dict.values() )
            for x in range(m):
                if alive_dict[x] == 1: # count number of sce outside Star(x) covered
                    num_alive_outside = num_alive 
                    for y in ECG[x]: # num. alive outside Star(x) = total num. of alive sce - num. alive in Star(x)
                        if alive_dict[y] == 1:
                            num_alive_outside -= 1

                    if num_alive_outside <= 0: # all sce outside Star(x) are covered
                        stop_flag_1 = 1 
            #print(num_alive_outside)

            if stop_flag_1 == 1:
                cover_time = t + 1 # need to plus 1 since t starts at 0
                break

    elif skip_stupid_test == 1:
        for t in range(len(perm)):
            if stop_flag_1 == 1:
                #print("Phase 1 ends after", cover_time, "tests")
                break
            not_stupid = 0 # need to check whether this test is stupid
            test_index = perm[t] 
            
            for x in alive_dict: # update alive_dict
                if alive_dict[x] == 1 and input_table[x][str(test_index)] != 'u': 
                    not_stupid = 1 # found evidence that the test is not stupid
                    
            if not_stupid == 1: # if it's a not stupid
                cover_time += 1
                
                for x in alive_dict: # update alive_dict
                    if input_table[x][str(test_index)] != 'u' and input_table[x][str(test_index)] != outcome[test_index]:
                        alive_dict[x] = 0
                    
                # check whether there is a star S s.t. \bar S is covered
                num_alive = sum(alive_dict.values() )
                for x in range(m): # for each Star(x), check whether itis good
                    if alive_dict[x] == 1: # count num- of sce outside Star(x) covered
                        num_alive_outside = num_alive
                        for y in ECG[x]:
                            if alive_dict[y] == 1: # shouldn't have been counted in num_alive_outside
                                num_alive_outside -= 1

                        if num_alive_outside <= 0: # all sce outside Star(x) are covered
                            stop_flag_1 = 1 
                #print(num_alive_outside)
                
    # now diam <= 2, but may not be a clique. Need to continue until it becomes a clique
    # print("phase 1 ends after", cover_time, "tests")
    while stop_flag_2 == 0:  
        cover_time += 1
        alive_list = []
        for x in alive_dict:
            if alive_dict[x] == 1:
                alive_list.append(x) # now we have a list of alive scenarios
        stop_flag_2 = 1 # if we find some x,y with no edge, set flag = 0 
        for x_temp in alive_list:
            for y_temp in alive_list:
                temp = (y_temp in ECG[x]) # whether y is a nb of x
                if temp == False:
                    stop_flag_2 = 0
                    x = x_temp
                    y = y_temp

        if stop_flag_2 == 0: # now search for a test separating x,y
            for T in range(n):
                if input_table[x][str(T)] != 'u' and input_table[y][str(T)] != 'u' and input_table[x][str(T)] != input_table[y][str(T)]:
                    break
            if outcome[T] != input_table[x][str(T)]:
                alive_dict[x] = 0
            else:
                alive_dict[y] = 0

    return cover_time


###### compute the cover time of a permutation
## equiva- class idetification, with 'neighborhood stopping criterion'
def compute_cover_time_ec_nbhd(perm, outcome, skip_stupid_test, ECG): 
# input: 
# perm = a permutation of tests
# outcome = a realized outcome vector, each entry 1/0
# skip_stupid_test: 0/1, whether or not skip a test which rules out nothing ('stupid test')
    cover_time = 0
    alive_dict = {i: 1 for i in range(m)} # keep track of the status of sce, 1 for alive, 0 for dead
    stop_flag = 0 # becomes 1 if there is a Star(x) s.t. all sce- outside are ruled out

    if skip_stupid_test == 0:
        for t in range(len(perm)):
            test_index = perm[t]
            for x in alive_dict: # update alive_dict
                if alive_dict[x] == 1 and outcome[test_index] != input_table[x][str(test_index)] and input_table[x][str(test_index)] != 'u': 
                    alive_dict[x] = 0 # rule out x

            # check whether there is a star S s.t. \bar S is covered
            num_alive = sum(alive_dict.values() ) # total num of alive, including those inside a star
            for x in range(m): # check Star(x)
                if alive_dict[x] == 1: # count num- of sce outside Star(x) covered
                    num_alive_outside = num_alive
                    for y in ECG[x]:
                        if alive_dict[y] == 1: # shouldn't have been counted in num_alive_outside
                            num_alive_outside -= 1

                    if num_alive_outside <= 0: # i.e. all scenarios outside Star(x) are dead 
                        stop_flag = 1        

            if stop_flag == 1: # if there are at most 1 alive, declare done
                cover_time = t + 1
                break


    elif skip_stupid_test == 1:
        for t in range(len(perm)):
            test_index = perm[t]
            not_stupid = 0
            for x in alive_dict: # update alive_dict
                if alive_dict[x] == 1 and input_table[x][str(test_index)] != 'u':
                    not_stupid = 1 # this test is not stupid, so perform it
                    break
                    
            if not_stupid == 1: # do the following only when the test is not stupid
                cover_time += 1
                for x in alive_dict: # update alive_dict
                    if input_table[x][str(test_index)] != 'u' and input_table[x][str(test_index)] != outcome[test_index]:
                        alive_dict[x] = 0

                # compute diameter of the remaining graph
                num_alive = sum(alive_dict.values() )
                for x in range(m): # for each x, check whether Star(x) is good 
                    if alive_dict[x] == 1: # count num- of sce outside Star(x) covered
                        num_alive_outside = num_alive
                        for y in ECG[x]:
                            if alive_dict[y] == 1: # shouldn't have been counted in num_alive_outside
                                num_alive_outside -= 1

                        if num_alive_outside <= 0: 
                            stop_flag = 1        

                if stop_flag == 1: # if there are at most 1 alive, declare done
                    cover_time = t + 1
                    break

    else: 
        print("error!")

    return cover_time

