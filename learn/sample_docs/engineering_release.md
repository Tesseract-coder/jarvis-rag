# Halcyon Robotics — Engineering Release Process

## Code review

Every change to the main branch goes through a pull request. A pull request needs two approvals before it can be merged, and at least one of them must come from a code owner of the affected directory. Authors may not approve their own pull requests. Continuous integration must be green; flaky tests are quarantined by the test-health rotation rather than retried until they pass.

## Release train

Robot firmware and the fleet management software ship on a release train every second Tuesday. Code freeze is on the Friday at 17:00 before the release. After the freeze only release-blocking fixes are merged, and each one needs sign-off from the release captain. The release captain role rotates between senior engineers each train.

## Canary and rollback

Each release is first deployed to a canary group of 5% of the customer fleet for 30 minutes. If the error rate of the canary group rises more than 0.5 percentage points above the baseline, the release is rolled back automatically. Manual rollback is done with `halctl rollback --to previous`.

## Hotfixes

A hotfix bypasses the release train and is only allowed for SEV1 and SEV2 incidents. The incident commander must approve every hotfix. Hotfixes are deployed with `halctl deploy --hotfix` and still go through the canary stage, shortened to 10 minutes.

## On-call

Each product team runs a one-week on-call rotation, with handover every Monday at 09:00. The on-call engineer must acknowledge a page within 5 minutes.

Incident severities:

- SEV1: customer robots stopped or unsafe. Response within 15 minutes, any time of day.
- SEV2: major feature degraded for many customers. Response within 1 hour.
- SEV3: minor issue or single customer affected. Handled the next working day.

Every SEV1 and SEV2 incident gets a blameless postmortem, published within 5 working days.
