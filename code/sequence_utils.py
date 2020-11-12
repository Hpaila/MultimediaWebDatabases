def get_cost(t1, t2, custom_cost):
    if custom_cost == False or custom_cost == None:
        return 1
    cost = 0
    for i in range(len(t1)):
        cost += abs(t1[i] - t2[i])
    return cost

def get_edit_distance(list1, list2, custom_cost = False):
    m = len(list1)
    n = len(list2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 

    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0: 
                dp[i][j] = j 
            elif j == 0: 
                dp[i][j] = i  
            elif list1[i-1] == list2[j-1]: 
                dp[i][j] = dp[i-1][j-1]
            else: 
                dp[i][j] = min(get_cost((0,0,0), list2[j-1], custom_cost) + dp[i][j-1],        
                                get_cost((0,0,0), list1[i-1], custom_cost) + dp[i-1][j],     
                                get_cost(list1[i-1], list2[j-1], custom_cost) + dp[i-1][j-1])   
    return dp[m][n]

def get_dtw_distance(list1, list2):
    m = len(list1)
    n = len(list2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
    for i in range(m + 1): 
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = 0
            elif j == 0:
                dp[i][j] = 0
            else:
                dp[i][j] = abs(list1[i-1] - list2[j-1]) + min(dp[i][j-1],        
                                   dp[i-1][j],     
                                   dp[i-1][j-1])   
    return dp[m][n]