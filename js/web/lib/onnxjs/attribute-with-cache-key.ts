// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

class AttributeWithCacheKeyImpl {
  constructor(attribute: Record<string, unknown>) {
    Object.assign(this, attribute);
  }

  private _cacheKey: string;
  public get cacheKey(): string {
    if (!this._cacheKey) {
      this._cacheKey =
          Object.getOwnPropertyNames(this).sort().map(name => `${(this as Record<string, unknown>)[name]}`).join(';');
    }
    return this._cacheKey;
  }
}

export type AttributeWithCacheKey<T> = T&{readonly cacheKey: string};

export const createAttributeWithCacheKey =
    <T extends Record<string, unknown>>(attribute: T): AttributeWithCacheKey<T> =>
        new AttributeWithCacheKeyImpl(attribute) as unknown as AttributeWithCacheKey<T>;
