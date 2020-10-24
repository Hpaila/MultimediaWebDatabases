def get_edit_distance(list1, list2):
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
                dp[i][j] = 1 + min(dp[i][j-1],        
                                   dp[i-1][j],     
                                   dp[i-1][j-1])   
    return dp[m][n]

def get_dtw_distance(list1, list2):
    m = len(list1)
    n = len(list2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)] 
    for i in range(m + 1): 
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = list2[j-1]
            elif j == 0:
                dp[i][j] = list1[i-1]
            else:
                dp[i][j] = abs(list1[i-1] - list2[j-1]) + min(dp[i][j-1],        
                                   dp[i-1][j],     
                                   dp[i-1][j-1])   
    return dp[m][n]