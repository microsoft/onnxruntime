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

export interface AttributeWithCacheKey {
  readonly cacheKey: string;
}

/**
 * create a new object from the given attribute, and add a cacheKey property to it
 */
export const createAttributeWithCacheKey = <T extends Record<string, unknown>>(attribute: T): T&AttributeWithCacheKey =>
    new AttributeWithCacheKeyImpl(attribute) as unknown as T & AttributeWithCacheKey;
