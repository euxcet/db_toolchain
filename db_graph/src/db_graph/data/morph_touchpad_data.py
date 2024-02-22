class MorphContactData():
  ''' Contains all info in one keystroke
      
  Attributes:
      id: a unique integer to number a keystroke with upper limit 16
      state: 1 => the start of a keystroke, 2 => middle, 3 => the end
          number of states that equal 1 or 3 can only be 1
      label: -1 => unsure, 0 => negative, 1 => positive
  '''

  def __init__(
      self,
      id = 0,
      state = 0,
      x = 0,
      y = 0,
      area = 0,
      force = 0,
      major = 0,
      minor = 0,
      delta_x = 0,
      delta_y = 0,
      delta_force = 0,
      delta_area = 0,
      label = 0
  ) -> None:
    self.id = id
    self.state = state
    self.x = x
    self.y = y
    self.area = area
    self.force = force
    self.major = major
    self.minor = minor
    self.delta_x = delta_x
    self.delta_y = delta_y
    self.delta_force = delta_force
    self.delta_area = delta_area
    self.label = label

  def __str__(self) -> str:
    return 'id: ' + str(self.id) + '\n' + 'state: ' + str(self.state) + '\n' \
      + 'x: ' + str(self.x) + '\n' + 'y: ' + str(self.y) + '\n' \
      +'area: ' + str(self.area) + '\n' + 'force: ' + str(self.force) + '\n' \
      + 'major: ' + str(self.major) + '\n' + 'minor: ' + str(self.minor) + '\n' \
      + 'delta_x: ' + str(self.delta_x) + '\n' + 'delta_y: ' + str(self.delta_y) + '\n' \
      + 'delta_force: ' + str(self.delta_force) + '\n' + 'delta_area: ' + str(self.delta_area) + '\n' \
      + 'label: ' + str(self.label)

class MorphTouchpadData():
  ''' info in one frame of morph board
  attrs:
      force_array: TODO
      contacts: list of keystroke contacts on the board
      timestamp: time acquired from time.perf_counter for each frame
  '''

  def __init__(self, force_array, timestamp):
    self.force_array = force_array
    self.timestamp = timestamp
    self.contacts = []
  
  def append_contact(self, contact):
    self.contacts.append(contact)