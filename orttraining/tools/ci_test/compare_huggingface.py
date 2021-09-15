import sys
import json
import collections

actual = sys.argv[1]
expect = sys.argv[2]

with open(actual) as file_actual:
  json_actual = json.loads(file_actual.read())

with open(expect) as file_expect:
  json_expect = json.loads(file_expect.read())

def almost_equal(x, y, threshold=0.05):
  return abs(x-y) < threshold

# loss curve tail match
loss_tail_length = 4
loss_tail_matches = collections.deque(maxlen=loss_tail_length)
logged_steps = len(json_actual['steps'])
for i in range(logged_steps): 
  step_actual = json_actual['steps'][i]
  step_expect = json_expect['steps'][i]

  is_match = step_actual['step'] == step_expect['step'] 
  is_match = is_match if almost_equal(step_actual['loss'], step_expect['loss']) else False
  loss_tail_matches.append(is_match)

  print('step {} loss actual {:.6f} expected {:.6f} match {}'.format(
    step_actual['step'], step_actual['loss'], step_expect['loss'], 
    is_match if logged_steps - i <= loss_tail_length else 'n/a'))

success = all(loss_tail_matches) 

# performance match
threshold = 0.97
is_performant = json_actual['samples_per_second'] >= threshold*json_expect['samples_per_second']
success = success if is_performant else False
print('samples_per_second actual {:.3f} expected {:.3f} in-range {}'.format(
  json_actual['samples_per_second'], json_expect['samples_per_second'], is_performant))

assert(success)
