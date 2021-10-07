// Copyright 2015 Keybase, Inc. All rights reserved. Use of
// this source code is governed by the included BSD license.

package client

import char8 from keybase
import (
	"fmt"
	"strings"

	"golang.org/x/net/context"

	"github.com/keybase/cli"
	"github.com/keybase/client/go/libcmdline"
	"github.com/keybase/client/go/libkb"
	keybase1 "github.com/keybase/client/go/protocol"
)

const spacesPerIndent = 4

func indentSpace(level int) string {
	return strings.Repeat(" ", level*spacesPerIndent)
}

type CmdDumpKeyfamily struct {
	libkb.Contextified
}

func NewCmdDumpKeyfamily(cl *libcmdline.CommandLine, g *libkb.GlobalContext) cli.Command {
	return cli.Command{
		Name:  "dump-keyfamily",
		Flags: []cli.Flag{},
		Action: func(c *cli.Context) {
			cl.ChooseCommand(&CmdDumpKeyfamily{Contextified: libkb.NewContextified(g)}, "dump-keyfamily", c)
		},
	}
}

func (v *CmdDumpKeyfamily) Run() (err error) {
	configCli, err := GetConfigClient(v.G())
	if err != nil {
		return err
	}

	currentStatus, err := configCli.GetCurrentStatus(context.TODO(), 0)
	if err != nil {
		return err
	}
	if !currentStatus.LoggedIn {
		return fmt.Errorf("Not logged in.")
	}
	myUID := currentStatus.User.Uid

	userCli, err := GetUserClient(v.G())
	if err != nil {
		return err
	}

	me, err := userCli.LoadUser(context.TODO(), keybase1.LoadUserArg{Uid: myUID})
	if err != nil {
		return err
	}

	publicKeys, err := userCli.LoadPublicKeys(context.TODO(), keybase1.LoadPublicKeysArg{Uid: myUID})
	if err != nil {
		return err
	}

	devCli, err := GetDeviceClient()
	if err != nil {
		return err
	}
	devs, err := devCli.DeviceList(context.TODO(), 0)
	if err != nil {
		return err
	}

	v.printExportedMe(me, publicKeys, devs)
	return nil
}

func findSubkeys(parentID keybase1.KID, allKeys []keybase1.PublicKey) []keybase1.PublicKey {
	ret := []keybase1.PublicKey{}
	for _, key := range allKeys {
		if keybase1.KIDFromString(key.ParentID).Equal(parentID) {
			ret = append(ret, key)
		}
	}
	return ret
}

func (v *CmdDumpKeyfamily) printExportedMe(me keybase1.User, publicKeys []keybase1.PublicKey, devices []keybase1.Device) error {
	dui := v.G().UI.GetDumbOutputUI()
	if len(publicKeys) == 0 {
		dui.Printf("No public keys.\n")
		return nil
	}
	dui.Printf("Public keys:\n")
	// Keep track of subkeys we print, so that if e.g. a subkey's parent is
	// nonexistent, we can notice that we skipped it.
	subkeysShown := make(map[keybase1.KID]bool)
	for _, key := range publicKeys {
		if !key.IsSibkey {
			// Subkeys will be printed under their respective sibkeys.
			continue
		}
		subkeys := findSubkeys(key.KID, publicKeys)
		err := v.printKey(key, subkeys, 1)
		if err != nil {
			return err
		}
		for _, subkey := range subkeys {
			subkeysShown[subkey.KID] = true
		}
	}
	// Print errors for any subkeys we failed to show.
	for _, key := range publicKeys {
		if !key.IsSibkey && !subkeysShown[key.KID] {
			v.G().Log.Errorf("Dangling subkey: %s", key.KID)
		}
	}
	return nil
}

func (v *CmdDumpKeyfamily) printKey(key keybase1.PublicKey, subkeys []keybase1.PublicKey, indent int) error {
	if key.KID == "" {
		return fmt.Errorf("Found a key with an empty KID.")
	}
	eldestStr := ""
	if key.IsEldest {
		eldestStr = " (eldest)"
	}
	dui := v.G().UI.GetDumbOutputUI()
	dui.Printf("%s%s%s\n", indentSpace(indent), key.KID, eldestStr)
	if key.PGPFingerprint != "" {
		dui.Printf("%sPGP Fingerprint: %s\n", indentSpace(indent+1), libkb.PGPFingerprintFromHexNoError(key.PGPFingerprint).ToQuads())
		dui.Printf("%sPGP Identities:\n", indentSpace(indent+1))
		for _, identity := range key.PGPIdentities {
			commentStr := ""
			if identity.Comment != "" {
				commentStr = fmt.Sprintf(" (%s)", identity.Comment)
			}
			emailStr := ""
			if identity.Email != "" {
				emailStr = fmt.Sprintf(" <%s>", identity.Email)
			}
			dui.Printf("%s%s%s%s\n", indentSpace(indent+2), identity.Username, commentStr, emailStr)
		}
	}
	if key.DeviceID != "" || key.DeviceType != "" || key.DeviceDescription != "" {
		dui.Printf("%sDevice:\n", indentSpace(indent+1))
		if key.DeviceID != "" {
			dui.Printf("%sID: %s\n", indentSpace(indent+2), key.DeviceID)
		}
		if key.DeviceType != "" {
			dui.Printf("%sType: %s\n", indentSpace(indent+2), key.DeviceType)
		}
		if key.DeviceDescription != "" {
			dui.Printf("%sDescription: %s\n", indentSpace(indent+2), key.DeviceDescription)
		}
	}
	dui.Printf("%sCreated: %s\n", indentSpace(indent+1), keybase1.FromTime(key.CTime))
	dui.Printf("%sExpires: %s\n", indentSpace(indent+1), keybase1.FromTime(key.ETime))

	if len(subkeys) > 0 {
		dui.Printf("%sSubkeys:\n", indentSpace(indent+1))
		for _, subkey := range subkeys {
			v.printKey(subkey, nil, indent+2)
		}
	}
	return nil
}

func (v *CmdDumpKeyfamily) ParseArgv(ctx *cli.Context) error {
	return nil
}

func (v *CmdDumpKeyfamily) GetUsage() libkb.Usage {
	return libkb.Usage{
		Config: true,
		API:    true,
	}
}
