// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

//
//  Simple CRITICAL_SECTION based locks
//
class CWinMLLock {
 private:
  // make copy constructor and assignment operator inaccessible

  CWinMLLock(const CWinMLLock& critical_section);
  CWinMLLock& operator=(const CWinMLLock& critical_section);

  CRITICAL_SECTION critical_section_;

 public:
  CWinMLLock() { InitializeCriticalSection(&critical_section_); };

  ~CWinMLLock() { DeleteCriticalSection(&critical_section_); };

  void Lock() { EnterCriticalSection(&critical_section_); };
  void Unlock() { LeaveCriticalSection(&critical_section_); };
  void LockExclusive() { EnterCriticalSection(&critical_section_); };
  void UnlockExclusive() { LeaveCriticalSection(&critical_section_); };
  BOOL IsLockHeldByCurrentThread() {
    return GetCurrentThreadId() == static_cast<DWORD>(reinterpret_cast<ULONG_PTR>(critical_section_.OwningThread));
  };
  BOOL IsLockHeld() { return critical_section_.OwningThread != 0; };
  BOOL TryLock() { return TryEnterCriticalSection(&critical_section_); };
  // aliased methods to help code compat so that CriticalSections can be passed to ReaderWriter templates
  void LockShared() { EnterCriticalSection(&critical_section_); };
  void UnlockShared() { LeaveCriticalSection(&critical_section_); };
};

// locks a critical section, and unlocks it automatically
// when the lock goes out of scope
class CWinMLAutoLock {
  // make copy constructor and assignment operator inaccessible

  CWinMLAutoLock(const CWinMLAutoLock& auto_lock);
  CWinMLAutoLock& operator=(const CWinMLAutoLock& auto_lock);

 protected:
  CWinMLLock* winml_lock_;

 public:
  CWinMLAutoLock(CWinMLLock* lock) {
    winml_lock_ = lock;
    if (winml_lock_ != nullptr) {
      winml_lock_->Lock();
    }
  };

  ~CWinMLAutoLock() {
    if (winml_lock_ != nullptr) {
      winml_lock_->Unlock();
    }
  };
};
