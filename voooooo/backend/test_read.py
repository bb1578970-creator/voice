import wfdb

record = wfdb.rdrecord("../dataset/voice001")

print(record)
print(record.p_signal.shape)