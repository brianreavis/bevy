use core::{any::type_name, fmt};

use crate::{
    entity::Entity,
    never::Never,
    system::{entity_command::EntityCommandError, Command, EntityCommand},
    world::{error::EntityMutableFetchError, World},
};

use super::{BevyError, ErrorContext, ErrorHandler};

/// Takes a [`Command`] that potentially returns a Result and uses a given error handler function to convert it into
/// a [`Command`] that internally handles an error if it occurs and returns `()`.
pub trait HandleError<Out = ()>: Send + 'static {
    /// Takes a [`Command`] that returns a Result and uses a given error handler function to convert it into
    /// a [`Command`] that internally handles an error if it occurs and returns `()`.
    ///
    /// If `None` is provided, the error will be completely ignored.
    fn handle_error_with(self, error_handler: Option<ErrorHandler>) -> impl Command;
    /// Takes a [`Command`] that returns a Result and uses the default error handler function to convert it into
    /// a [`Command`] that internally handles an error if it occurs and returns `()`.
    fn handle_error(self) -> impl Command;
}

impl<C, T, E> HandleError<Result<T, E>> for C
where
    C: Command<Result<T, E>>,
    E: Into<BevyError>,
{
    fn handle_error_with(self, error_handler: Option<ErrorHandler>) -> impl Command {
        move |world: &mut World| {
            let result = self.apply(world);
            if let Some(error_handler) = error_handler {
                match result {
                    Ok(_) => {}
                    Err(err) => (error_handler)(
                        err.into(),
                        ErrorContext::Command {
                            name: type_name::<C>().into(),
                        },
                    ),
                }
            } else {
                // If there's no error handler, we intentionally swallow the error
            }
        }
    }

    fn handle_error(self) -> impl Command {
        move |world: &mut World| match self.apply(world) {
            Ok(_) => {}
            Err(err) => world.default_error_handler()(
                err.into(),
                ErrorContext::Command {
                    name: type_name::<C>().into(),
                },
            ),
        }
    }
}

impl<C> HandleError<Never> for C
where
    C: Command<Never>,
{
    fn handle_error_with(
        self,
        _error_handler: Option<fn(BevyError, ErrorContext)>,
    ) -> impl Command {
        move |world: &mut World| {
            self.apply(world);
        }
    }

    #[inline]
    fn handle_error(self) -> impl Command {
        move |world: &mut World| {
            self.apply(world);
        }
    }
}

impl<C> HandleError for C
where
    C: Command,
{
    #[inline]
    fn handle_error_with(
        self,
        _error_handler: Option<fn(BevyError, ErrorContext)>,
    ) -> impl Command {
        self
    }
    #[inline]
    fn handle_error(self) -> impl Command {
        self
    }
}

/// Passes in a specific entity to an [`EntityCommand`], resulting in a [`Command`] that
/// internally runs the [`EntityCommand`] on that entity.
///
// NOTE: This is a separate trait from `EntityCommand` because "result-returning entity commands" and
// "non-result returning entity commands" require different implementations, so they cannot be automatically
// implemented. And this isn't the type of implementation that we want to thrust on people implementing
// EntityCommand.
pub trait CommandWithEntity<Out> {
    /// Passes in a specific entity to an [`EntityCommand`], resulting in a [`Command`] that
    /// internally runs the [`EntityCommand`] on that entity.
    fn with_entity(self, entity: Entity) -> impl Command<Out> + HandleError<Out>;
}

impl<C> CommandWithEntity<Result<(), EntityMutableFetchError>> for C
where
    C: EntityCommand,
{
    fn with_entity(
        self,
        entity: Entity,
    ) -> impl Command<Result<(), EntityMutableFetchError>>
           + HandleError<Result<(), EntityMutableFetchError>> {
        move |world: &mut World| -> Result<(), EntityMutableFetchError> {
            let entity = world.get_entity_mut(entity)?;
            self.apply(entity);
            Ok(())
        }
    }
}

impl<C, T, Err> CommandWithEntity<Result<T, EntityCommandError<Err>>> for C
where
    C: EntityCommand<Result<T, Err>>,
    Err: fmt::Debug + fmt::Display + Send + Sync + 'static,
{
    fn with_entity(
        self,
        entity: Entity,
    ) -> impl Command<Result<T, EntityCommandError<Err>>> + HandleError<Result<T, EntityCommandError<Err>>>
    {
        move |world: &mut World| {
            let entity = world.get_entity_mut(entity)?;
            self.apply(entity)
                .map_err(EntityCommandError::CommandFailed)
        }
    }
}
