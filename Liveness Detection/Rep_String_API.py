import time

class longestDupSubstring:

    def __init__(self, Eye_State):
        self.s = Eye_State
    
    
    def find(self):
        N = len(self.s)
        suffix = []
        for i in range(0,N):
            suffix.append(self.s[N-i-1:N])
        
        suffix=sorted(suffix)
        
        lrs=""
        length=0
        for i in range(0,N-1):
            length=self.lcp(suffix[i],suffix[i+1],len(lrs))
            
            if(length>len(lrs)):
                lrs=suffix[i][0:length]
                
        return lrs
    

    def lcp(self,s1,s2,current_len):
        if(len(s1)<len(s2)):
            limit=len(s1)
        else:
            limit=len(s2)
    
        
        if(s1[0:limit]==s2[0:limit]): # if substring are the same at limit, return limit
            return limit
        
        if(limit < current_len):      # if the limit is less than the length of current duplicated substring, we don't need to 
            return 0                  # compare.
        else:
            n = current_len
            while(s1[0:n+1]==s2[0:n+1] and n<=limit):
                n+=1
            
            if(n>current_len):
                return n
            else:
                return 0   
           
    def Rep_Detect_Check(self):
        repblink = False
        if(len(self.s) > 10):
            L_Rep = self.find()
            
            if(len(L_Rep) > 0) & (L_Rep != ('0'*len(L_Rep))) & (L_Rep != ('1'*len(L_Rep))):
            #( RepString's len > 0 ) and ( not build up by all-'0' or all-'1' ) => Fake
            
                print("\nEYE STATE:",self.s)
                print("Find Longest RepString:",L_Rep)
                print(">>Fake Detect!!!<<")
                repblink = True
                
                self.s = ''
                
        return self.s, repblink
                