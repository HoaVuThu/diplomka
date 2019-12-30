# LUBIM KRISTIANKA 
import json
import codecs

# with open('drg.json', encoding="utf8") as file:
  # s = file.read()
  # print(s[194:196])
s = codecs.open('E:\data\\drg\\drg-without-comma.json', 'r', 'utf-8-sig').read()
print(s[1332767712:1332767714])
datastore = json.loads(s)

# f = codecs.open('E:\\data\\drg\\drg.json', 'r', 'utf-8-sig')
# s = f.read()
# l = list(s)
# l[len(s)-2] = ''
# result = "".join(l)
# print(result[len(result)-2:])
# codecs.open('E:\\data\\drg\\drg-without-comma.json', 'w', 'utf-8-sig').write(result)