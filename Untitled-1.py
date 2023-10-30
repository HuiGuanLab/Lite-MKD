class Solution:
    def strongPasswordCheckerII(self, password: str) -> bool:
        print(password)
        n = len(password)
        if n < 8 or n > 22:
            return False
        check = [False] * 4
        for i in range(n):
            # print(i)
            ch = password[i]
            if i < n-2 and ch == password[i+1] and ch == password[i+2]:
                print('weak')
            if 97 <= ord(ch) <= 122:# ch是小写英文字母
                check[0] = True
            elif 65 <= ord(ch) <= 90:
                check[1] = True
            elif ch.isdigit():
                check[2] = True
            elif ch in"0123456789": # ch是特殊符号
                check[3] = True
        
        if all(check):
            print('strong')
        else:
            print('weak')
        # 若check中四个元素均为True，返回True
        
if __name__ == "__main__":
        
    sln = Solution()
    print(sln.strongPasswordCheckerII("1234567890Abcd"))
    print(sln.strongPasswordCheckerII("1234567890aaaa"))
    print(sln.strongPasswordCheckerII("123456789Gabcd"))




